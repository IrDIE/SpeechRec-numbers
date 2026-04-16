"""Audio augmentation pipeline: waveform-level (audiomentations) + SpecAugment."""
import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import FrequencyMasking, TimeMasking


# ---------------------------------------------------------------------------
# Waveform augmentation
# ---------------------------------------------------------------------------

class WaveformAugmentor:
    """Waveform-level augmentation via audiomentations.

    Applied to raw audio (numpy float32, shape (samples,)) before mel extraction,
    so it simulates acoustic variation that the spectrogram-level cannot capture
    (speaker pitch, room reverb, recording gain, speaking rate).
    """

    def __init__(self, sample_rate: int = 16000, p: float = 1.0):
        try:
            import audiomentations as A
        except ImportError as e:
            raise ImportError(
                "audiomentations is required for waveform augmentation. "
                "Install it with: pip install audiomentations"
            ) from e

        transforms = [
            # Random gain ±6 dB — simulates different microphone levels
            A.Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.4),
            # Low-level Gaussian noise — microphone self-noise
            A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            # Time stretch without changing pitch — speaking rate variation
            A.TimeStretch(min_rate=0.85, max_rate=1.15, leave_length_unchanged=False, p=0.3),
            # Pitch shift ±2 semitones — speaker pitch variation
            A.PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.3),
        ]

        # RoomSimulator requires pyroomacoustics; add only if available
        try:
            import pyroomacoustics  # noqa: F401
            transforms.append(A.RoomSimulator(p=0.2))
        except ImportError:
            pass

        self.augment = A.Compose(transforms, p=p)
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Augment a mono waveform (float32 numpy array)."""
        augmented = self.augment(samples=audio, sample_rate=self.sample_rate)
        # Renormalize to [-1, 1] to avoid clipping after augmentation
        max_val = np.abs(augmented).max()
        if max_val > 0:
            augmented = augmented / max_val
        return augmented.astype(np.float32)


# ---------------------------------------------------------------------------
# SpecAugment (via torchaudio)
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    """SpecAugment using torchaudio's FrequencyMasking and TimeMasking.

    torchaudio expects input shape (freq, time), while our mel is (time, n_mels),
    so we transpose before applying and transpose back after.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 70,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [TimeMasking(time_mask_param=time_mask_param) for _ in range(n_time_masks)]
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (time, n_mels)
        Returns:
            Augmented tensor of the same shape.
        """
        # torchaudio transforms expect (freq, time) — transpose in/out
        x = mel.T.unsqueeze(0)  # (1, n_mels, time)
        for mask in self.freq_masks:
            x = mask(x)
        for mask in self.time_masks:
            x = mask(x)
        return x.squeeze(0).T  # back to (time, n_mels)


# ---------------------------------------------------------------------------
# Combined pipeline factory
# ---------------------------------------------------------------------------

def build_train_augmentation(
    sample_rate: int = 16000,
    waveform_p: float = 1.0,
    freq_mask_param: int = 27,
    time_mask_param: int = 70,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> tuple["WaveformAugmentor", "SpecAugment"]:
    """Convenience factory that returns (waveform_augmentor, spec_augmentor).

    Both are applied only during training. Pass waveform_augmentor to
    RussianSpeechDataset so it runs *before* mel extraction (no cache).
    Pass spec_augmentor to the same dataset; it runs *after* mel extraction.
    """
    waveform_aug = WaveformAugmentor(sample_rate=sample_rate, p=waveform_p)
    spec_aug = SpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        n_freq_masks=n_freq_masks,
        n_time_masks=n_time_masks,
    )
    return waveform_aug, spec_aug
