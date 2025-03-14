# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Sequence, Any

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    TDSGRUEncoder,
    TDSConvEncoderv2,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

class TDSConvGRUCTCModulev2(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model (Conv, GRU model)
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            #####################
            # EDIT
            #####################
            # vanilla. Maxpooling does not work (kills the gradent)
            TDSConvEncoderv2(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
                activation=nn.LeakyReLU(),
                max_pooling=False,
                batch_norm=True,
            ),

            #GRU
            TDSGRUEncoder(
               num_features = num_features,
               #gru_hidden_size = 128,
               #num_gru_layers = 4,
               gru_hidden_size = 512,
               num_gru_layers = 5,
            ),

            # LSTM
            # TDSLSTMEncoder(
            #   input_dim=num_features,
            #   hidden_dim=64,
            #   output_dim=128,
            #   n_layers=4,
            # ),

            # TCN
            # TCNEncoder(
            #     num_features=num_features,
            #     num_channels=(128, 128),
            #     kernel_width=3,
            #     dropout=0.2,
            #     activation=nn.LeakyReLU,
            #     bn=True
            # ),
            ####################
            # EDIT. REMEMBER THE COMMA
            ####################
            # Dropout rate 0.2 added
            nn.Dropout(0.2),
            # CHANGE HERE
            # (T, N, num_classes). nn.Linear(output_dim (from LSTM, TCN, RNN), charset().num_classes).
            # vanilla & GRU is num_features
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

class TDSConvGRUCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            TDSGRUEncoder(
                num_features = num_features,
                gru_hidden_size = 128,
                num_gru_layers = 4
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

class CNN_LSTM_MLP_Spec_CTCModule(pl.LightningModule):
    """
    CNN + LSTM + CTC model that preserves frequency information by applying the 
    rotation invariant MLP per frequency bin.
    
    Data flow:
      1. Input of shape (T, N, bands, electrode_channels, freq) is normalized via SpectrogramNorm.
      2. For each frequency bin, we extract a slice of shape (T, N, bands, electrode_channels) 
         and pass it through a MultiBandRotationInvariantMLP. This produces an output of shape 
         (T, N, bands, mlp_features[-1]) for that frequency.
      3. We stack these outputs over the frequency dimension, resulting in a tensor of shape 
         (T, N, bands, mlp_features[-1], freq).
      4. We then permute and reshape this tensor to form a CNN input of shape 
         (N, bands * mlp_features[-1], T, freq).
      5. A CNN stack with kernel size (3,3) (and padding (1,1)) processes this tensor, preserving 
         both time and frequency dimensions. Its output is of shape (N, final_cnn_channels, T, freq).
      6. We flatten the channel and frequency dimensions (to capture spectral features) so that the LSTM 
         receives an input of shape (T, N, final_cnn_channels * freq).
      7. The LSTM (bidirectional) processes the sequence, and its output is passed through dropout 
         and a fully connected layer (with LogSoftmax) to produce per-timestep class log probabilities.
    
    Assumptions:
      - Input tensor shape: (T, N, bands, electrode_channels, freq), where typically bands=2 and electrode_channels=16.
      - The SpectrogramNorm expects channels = bands * electrode_channels.
      - For each frequency bin, the MultiBandRotationInvariantMLP is applied on a slice of shape (T, N, bands, electrode_channels).
      - The output of MultiBandRotationInvariantMLP (per frequency) is (T, N, bands, mlp_features[-1]).
      - After stacking over freq, we assume the CNN can process the resulting tensor with kernel (3,3).
    """
    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        cnn_out_channels: list[int],
        mlp_in_features: int,      # Expected to be equal to electrode_channels (e.g., 16)
        mlp_features: list[int],   # Defines the hidden layers for the MLP; output dimension = mlp_features[-1]
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        spec_freq_bins: int,       # Number of frequency bins (e.g., n_fft//2+1, typically 33)
        dropout_rate: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # --- Pre-CNN Layers ---
        # 1. SpectrogramNorm is applied to the entire input. See modules.py for implementation.
        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        
        # 2. We will use a MultiBandRotationInvariantMLP (from modules.py) to process each frequency bin separately.
        #    Note: The built-in MBRMLP expects input shape (T, N, num_bands, in_features).
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=mlp_in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS
        )
        
        # --- CNN Stack ---
        # After applying the MLP to each frequency bin, we expect to obtain a tensor of shape:
        # (T, N, bands, mlp_features[-1], freq)
        # We then want to combine the bands and mlp_features dimensions into a channel dimension.
        in_channels = self.NUM_BANDS * mlp_features[-1]
        cnn_layers = []
        # Use kernel size (3,3) to process both temporal and frequency dimensions.
        for out_channels in cnn_out_channels:
            cnn_layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        # --- LSTM and Final Layers ---
        # After CNN, we expect an output tensor of shape (N, final_cnn_channels, T, freq).
        # We then flatten the channels and frequency dimensions to form LSTM input:
        # Shape becomes (T, N, final_cnn_channels * freq)
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels[-1] * spec_freq_bins,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )
        
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def apply_mlp_preserve_freq(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the MultiBandRotationInvariantMLP to preserve frequency information.
        
        Args:
          x: Input tensor of shape (T, N, bands, electrode_channels, freq)
        
        Returns:
          Tensor of shape (T, N, bands, mlp_features[-1], freq)
        """
        T, N, bands, _, freq = x.shape
        outputs = []
        for i in range(freq):
            # Extract slice for frequency bin i.
            # Shape becomes (T, N, bands, electrode_channels)
            slice_i = x[..., i]
            # Apply the MultiBandRotationInvariantMLP on this slice.
            # According to modules.py, the MBRMLP expects input shape (T, N, bands, in_features)
            # and returns (T, N, bands, mlp_features[-1]).
            out_i = self.mlp(slice_i)
            outputs.append(out_i)
        # Stack along a new frequency dimension.
        # Final shape: (T, N, bands, mlp_features[-1], freq)
        return torch.stack(outputs, dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
          inputs: Tensor of shape (T, N, bands, electrode_channels, freq)
        
        Returns:
          Logits of shape (T, N, num_classes)
        """
        # 1. Apply spectrogram normalization.
        x = self.spec_norm(inputs)  # Shape remains (T, N, bands, electrode_channels, freq)
        
        # 2. Apply MLP to each frequency bin to preserve frequency resolution.
        x = self.apply_mlp_preserve_freq(x)  # Shape: (T, N, bands, mlp_features[-1], freq)
        
        # 3. Permute and reshape to prepare for the CNN.
        # From (T, N, bands, mlp_features[-1], freq) to (N, bands * mlp_features[-1], T, freq)
        x = x.permute(1, 2, 3, 0, 4)  # Now (N, bands, mlp_features[-1], T, freq)
        N = x.shape[0]
        x = x.reshape(N, -1, x.shape[3], x.shape[4])
        
        # 4. Apply CNN stack.
        # Expected output shape: (N, final_cnn_channels, T, freq)
        x = self.cnn(x)
        
        # 5. Flatten the channel and frequency dimensions for LSTM.
        # Reshape from (N, final_cnn_channels, T, freq) to (N, final_cnn_channels * freq, T)
        if x.ndim == 4:
            N, C, T, F = x.shape
            x = x.view(N, C * F, T)
        else:
            raise ValueError("CNN output does not have the expected 4 dimensions.")
        
        # 6. Permute to (T, N, C * F) for LSTM.
        x = x.permute(2, 0, 1)
        
        # 7. Pass through LSTM.
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # 8. Final fully connected layer to produce log probabilities.
        logits = self.fc(lstm_out)
        return logits

    # The remaining training, validation, test, and optimizer methods are copied verbatim.
    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]
        input_lengths, target_lengths = batch["input_lengths"], batch["target_lengths"]
        N = len(input_lengths)
        emissions = self.forward(inputs)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff
        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("test", batch)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class CNN_LSTM_CTCModule(pl.LightningModule):
    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        cnn_out_channels: list[int],
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        dropout_rate: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        """
        Initializes the CNN+LSTM model with an arbitrary number of convolutional layers.
        
        Assumptions:
         - `cnn_out_channels` is a list of integers, where each integer specifies the
           number of output channels for one CNN layer.
         - The input tensor has shape (T, N, bands, electrode_channels, freq),
           where T is the number of time steps, N is the batch size, `bands` equals NUM_BANDS
           (typically 2), `electrode_channels` equals 16, and `freq` is the number of frequency bins.
         - The CNN will operate on an input reshaped to (N, NUM_BANDS * electrode_channels, T, freq).
         - After the CNN stack, we average over the frequency dimension before passing the result
           to the LSTM. The LSTM input size is therefore equal to the last element in cnn_out_channels.
        """
        super().__init__()
        self.save_hyperparameters()

        # Build CNN stack as a Sequential module.
        # Initial input channels: NUM_BANDS * ELECTRODE_CHANNELS (e.g., 2*16 = 32)
        in_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS
        layers = []
        for out_channels in cnn_out_channels:
            layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)  # Dropout after activation to regularize.
            ])
            # Update in_channels for the next layer
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        # LSTM for temporal modeling.
        # We assume that after the CNN, the output tensor has shape (N, last_cnn_out_channels, T, F),
        # where F is the frequency dimension. We average over F to get a tensor of shape (N, last_cnn_out_channels, T).
        # Then we permute it to (T, N, last_cnn_out_channels) before passing it to the LSTM.
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=False,  # expects input shape (T, N, feature_dim)
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )

        # Extra dropout layer after LSTM.
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer to output log-probabilities over character classes.
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # CTC Loss: using the blank label from our character set.
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Instantiate decoder from configuration.
        self.decoder = instantiate(decoder)

        # Metrics (Character Error Rates).
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN + LSTM model.
        
        Assumptions:
         - `inputs` is a tensor of shape (T, N, bands, electrode_channels, freq),
           where T is the number of time steps, N is the batch size, bands equals NUM_BANDS,
           electrode_channels equals 16, and freq is the number of frequency bins.
        """
        T, N, B, C, F = inputs.shape  # B should equal NUM_BANDS and C should equal 16.
        # Permute and reshape to combine bands and channels:
        # (T, N, B, C, F) -> (N, B * C, T, F)
        x = inputs.permute(1, 2, 3, 0, 4).reshape(N, B * C, T, F)
        
        # Pass through the CNN stack.
        # Output shape becomes (N, last_cnn_out_channels, T, F)
        x = self.cnn(x)
        
        # Average over the frequency dimension to collapse F.
        # Resulting shape: (N, last_cnn_out_channels, T)
        x = torch.mean(x, dim=3)
        
        # Permute to (T, N, last_cnn_out_channels) for LSTM.
        x = x.permute(2, 0, 1)
        
        # LSTM forward pass.
        # LSTM output shape: (T, N, lstm_hidden_dim * 2) due to bidirectionality.
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout on the LSTM output before the fully connected layer.
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layer to obtain log probabilities.
        logits = self.fc(lstm_out)  # Shape: (T, N, num_classes)
        return logits

    # Below, all other functions are copied exactly from TDSConvCTCModule.

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]
        input_lengths, target_lengths = batch["input_lengths"], batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # T_diff accounts for any reduction in the temporal dimension
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()

        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("test", batch)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class CNN_AE_GRU_CTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_channels: Sequence[int],
        kernel_width: int,
        ae_features: Sequence[int],
        ae_dropout: float,
        gru_hidden_dim: int,
        gru_layers: int,
        optimizer: dict,
        lr_scheduler: dict,
        decoder: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.spectrogram_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )

        cnn_layers = []
        in_channels = self.NUM_BANDS  # correct initial channels after unflattening
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_width), stride=(1,1), padding=(0,kernel_width//2)))
            cnn_layers.append(nn.BatchNorm2d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(0.1))
            in_channels = out_channels

        cnn_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        cnn_layers.append(nn.Flatten(start_dim=1))
        self.cnn = nn.Sequential(*cnn_layers)

        self.encoder = nn.Sequential(
            nn.Linear(cnn_channels[-1], ae_features[0]),
            nn.ReLU(),
            nn.Dropout(ae_dropout),
            nn.Linear(ae_features[0], ae_features[1]),
            nn.ReLU(),
            nn.Dropout(ae_dropout),
        )

        self.gru = nn.GRU(
            input_size=ae_features[1],
            #hidden_size=ae_features[0],
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            dropout=0.4,
            batch_first=False,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 512),  # Extra layer before softmax
            nn.ReLU(),
            nn.Linear(512, charset().num_classes),
            nn.LogSoftmax(dim=-1),
            #nn.Linear(gru_hidden_dim * 2, charset().num_classes),
            #nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (T, N, B, C, freq)
        T, N, B, C, freq = inputs.shape
        x = self.spectrogram_norm(inputs)
        x = self.mlp(x)  # (T, N, B, mlp_features[-1])
        x = x.flatten(start_dim=2)  # (T, N, B * mlp_features[-1])

        
        #x = x.permute(1, 2, 0).reshape(N, B, -1, T)
        x = x.permute(1,2,0).reshape(N, self.NUM_BANDS, -1, T)

        x = self.cnn(x)  # CNN should now correctly accept this shape
        x = self.encoder(x)
        x = x.unsqueeze(0).repeat(T, 1, 1)
        gru_out, _ = self.gru(x)

        logits = self.classifier(gru_out)
        return logits


    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )



class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
