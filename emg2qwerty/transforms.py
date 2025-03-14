# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio
from scipy import signal


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
        sample_rate (int): keep track of the sampling rate for proper application
        of fourier transform
    """

    n_fft: int = 64
    hop_length: int = 16
    sample_rate: int = 2000

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )
        # Initialize the bandpass filter
        low_freq = 30 #Hz
        high_freq = 1000 #Hz
        self.bandpass_filter = BandpassFilter(n_fft=self.n_fft, low_freq=low_freq, high_freq=high_freq, sample_rate=self.sample_rate)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)

        #apply bandpass filter
        filtered_spec = self.bandpass_filter.apply(spec)

        logspec = torch.log10(filtered_spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)

#Eric's custom functions start here
@dataclass
class LogSpectrogramArbBase:
    """Creates a spectrogram with a configurable logarithmic base.

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins. (default: 64)
        hop_length (int): Number of samples to stride between consecutive STFT windows. (default: 16)
        log_base (float): The logarithm base to use. Must be greater than 1. (default: 10)
    """

    n_fft: int = 64
    hop_length: int = 16
    log_base: float = 10  # Default to log10

    def __post_init__(self) -> None:
        if self.log_base <= 1:
            raise ValueError("log_base must be greater than 1.")
        
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        
        # Apply logarithmic scaling based on the chosen base
        logspec = torch.log10(spec + 1e-6) / torch.log10(torch.tensor(self.log_base, dtype=spec.dtype))
        
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)

@dataclass
class TimeWarp:
    """
    Warps the signal along the time axis.
    
    Expects an input tensor of shape (time, field, electrode_channel). 
    A random warp factor is chosen and the signal is re-sampled along the time dimension.
    """
    def __init__(self, max_time_warp=0.2):
        self.max_time_warp = max_time_warp

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Get the number of time steps from the first dimension
        time_steps = tensor.shape[0]
        # Choose a warp factor between 1 - max_time_warp and 1 + max_time_warp
        warp_factor = 1.0 + np.random.uniform(-self.max_time_warp, self.max_time_warp)
        # Create new time indices by scaling the original indices
        original_indices = np.arange(time_steps)
        new_indices = np.linspace(0, time_steps - 1, time_steps) * warp_factor
        # Ensure new indices are within valid range
        new_indices = np.clip(new_indices, 0, time_steps - 1)
        
        # Convert tensor to numpy array (shape: time, field, electrode_channel)
        tensor_np = tensor.cpu().numpy()
        warped = np.empty_like(tensor_np)
        # Interpolate along the time axis for each field and channel
        for f in range(tensor_np.shape[1]):
            for c in range(tensor_np.shape[2]):
                warped[:, f, c] = np.interp(new_indices, original_indices, tensor_np[:, f, c])
        return torch.from_numpy(warped).to(tensor.device)

@dataclass
class AddGaussianNoise:
    """
    Adds Gaussian noise independently to each element.
    
    For an input tensor of shape (time, signal, electrode_channel), this function
    first computes and prints the average value over the time and channel dimensions (per signal),
    then adds independent Gaussian noise to each element using the specified mean and standard deviation.
    """
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Compute the average over the time and channel dimensions for each signal (field)
        #avg = tensor.mean(dim=(0, 2))
        #print("Average over time and channel per signal:", avg)
        # Generate noise independently for each element
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return tensor + noise

@dataclass
class AmplitudeRandomScaling:
    """
    Scales the amplitude of the signal along the field dimension.
    
    For an input tensor of shape (time, field, electrode_channel), a unique scaling factor
    is generated for each field and then broadcast over the time and channel dimensions.
    """
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Generate one scaling factor per field
        scaling_factors = torch.empty(tensor.shape[1], device=tensor.device).uniform_(*self.scale_range)
        # Reshape to (1, field, 1) so it broadcasts over time and channel dimensions
        scaling_factors = scaling_factors.view(1, -1, 1)
        return tensor * scaling_factors

@dataclass
class ChannelDropout:
    """
    Simulates electrode dropout by zeroing out one electrode channel per signal.
    
    For an input tensor of shape (time, signal, electrode_channel), this function
    randomly selects one channel (i.e. one of the 16 channels) for each signal and sets 
    its values across all time steps to zero, effectively simulating the dropout of an electrode.
    
    The dropout_prob parameter (defaulting to 1.0) controls the probability of applying the dropout 
    to each signal. With a probability of 1.0, every signal will have one channel dropped.
    """
    def __init__(self, dropout_prob=1.0):
        self.dropout_prob = dropout_prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Clone the tensor to avoid in-place modifications
        tensor_out = tensor.clone()
        _, num_signals, num_channels = tensor.shape
        # For each signal (the 1st axis)
        for s in range(num_signals):
            if np.random.rand() < self.dropout_prob:
                # Randomly choose one electrode channel to drop
                channel_to_drop = np.random.randint(num_channels)
                # Set the entire time series for that channel to 0
                tensor_out[:, s, channel_to_drop] = 0.0
        return tensor_out

@dataclass
class AmplitudeScaling:
    """Scales the amplitude of the specified fields in a numpy structured array.

    Args:
        fields (list): List of field names to be scaled.
        scale_factor (float): The factor by which to scale the amplitude.
            If None, each field will be scaled to have a maximum absolute
            value of 1.
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    scale_factor: float = None

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.scale_factor is None:
            for field in self.fields:
                max_abs = np.max(np.abs(data[field]))
                if max_abs == 0:
                    continue #prevent divide by zero
                data[field] = data[field] / max_abs
        else:
            for field in self.fields:
                data[field] = data[field] * self.scale_factor
        return data

@dataclass
class BandpassFilter:
    """
      Args:
        n_fft (int): size of FFT, creates n_fft // 2 + 1 frequency bins
        low_freq (float): the parameter to determine the lowest frequency
        passed through filter
        high_freq (float): the parameter to determine the highest frequency
        passed through filter
    """
    def __init__(self, n_fft: int, low_freq: float, high_freq: float, sample_rate: int) -> None:
        self.n_fft = n_fft
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sample_rate = sample_rate

        #create the frequency array for the bins
        self.freqs = torch.fft.fftfreq(self.n_fft, d=1.0 / self.sample_rate)[:self.n_fft // 2 + 1]
    def apply(self, spectrogram: torch.Tensor) -> torch.Tensor:
        #apply the filter
        low_idx = torch.searchsorted(self.freqs, self.low_freq)
        high_idx = torch.searchsorted(self.freqs, self.high_freq)

        #reduce the spectrogram only to the bins within the filter:
        spectrogram[..., :low_idx, :] = 0 #gets rid of frequencies lower than the filter
        spectrogram[..., high_idx:, :] = 0 #gets rid of frequencies higher than the filter

        return spectrogram

@dataclass
class BandpassPreFilter:
    """Applies a bandpass filter to the EMG signal fields in a numpy structured array.

    Args:
        lowcut (float): Lower cutoff frequency (Hz).
        highcut (float): Upper cutoff frequency (Hz).
        fs (float): Sampling rate (Hz).
        order (int): Order of the filter.
        fields (list): list of fields to apply the filter to.
    """

    lowcut: float = 5.0
    highcut: float = 500.0
    fs: float = 2000.0
    order: int = 4
    fields: list = ("emg_left", "emg_right")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = signal.butter(self.order, [low, high], btype="band")

        for field in self.fields:
            data[field] = signal.filtfilt(b, a, data[field], axis=0)

        return data
