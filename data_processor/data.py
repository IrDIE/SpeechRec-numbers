from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from .postprocessor import RussianCharTokenizer, RussianWordTokenizer, NumericTokenizer


def load_audio(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file, resample to target_sr, normalize to [-1, 1]."""
    try:
        waveform, sr = torchaudio.load(str(path))
        audio = waveform.squeeze().numpy()
    except Exception:
        audio, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio.astype(np.float32)


def build_tokenizer(tokenizer_type: str):
    if tokenizer_type == "char":
        return RussianCharTokenizer()
    if tokenizer_type == "numeric":
        return NumericTokenizer()
    return RussianWordTokenizer()


class BaseSpeechDataset(Dataset):
    """Audio loading + mel feature extraction. Sufficient for test/inference."""

    def __init__(self, data_root, csv_path, audio_subdir, target_sr=16000,
                 n_mels=80, hop_length=200, n_fft=400, win_length=400):
        self.data_root = Path(data_root)
        self.target_sr = target_sr

        if csv_path is None:
            csv_path = self.data_root / f"{audio_subdir}.csv"
        self.df = pd.read_csv(csv_path, sep=",")

        self.audio_paths = []
        for fn in self.df["filename"]:
            if "/" in str(fn) or "\\" in str(fn):
                self.audio_paths.append(self.data_root / fn)
            else:
                self.audio_paths.append(self.data_root / audio_subdir / fn)

        self._mel = MelSpectrogram(
            sample_rate=target_sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, win_length=win_length, f_min=0.0, f_max=8000.0, norm=None,
        )

    def _extract(self, audio: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(audio)
        mel = self._mel(wav)             # (n_mels, time)
        return torch.log(mel + 1e-6).T  # (time, n_mels)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio = load_audio(self.audio_paths[idx], self.target_sr)
        features = self._extract(audio)
        return {
            "features": features,
            "feature_length": features.shape[0],
            "filename": str(self.df.iloc[idx]["filename"]),
        }

    def collate_fn(self, batch):
        feat_lens = [b["feature_length"] for b in batch]
        max_feat_len = max(feat_lens)
        n_mels = batch[0]["features"].shape[1]
        padded = torch.zeros(len(batch), max_feat_len, n_mels)
        for i, b in enumerate(batch):
            padded[i, : b["feature_length"]] = b["features"]
        return {
            "features": padded,
            "feature_lengths": torch.tensor(feat_lens, dtype=torch.long),
            "filenames": [b["filename"] for b in batch],
        }


class RussianSpeechDataset(BaseSpeechDataset):
    """Adds label encoding, RAM preloading, and speaker metadata for train/val."""

    def __init__(self, data_root, csv_path, tokenizer, audio_subdir,
                 target_sr=16000, n_mels=80, hop_length=200, n_fft=400, win_length=400,
                 waveform_augmentor=None, spec_augmentor=None):
        super().__init__(data_root, csv_path, audio_subdir, target_sr, n_mels, hop_length, n_fft, win_length)
        self.tokenizer = tokenizer
        self.waveform_augmentor = waveform_augmentor
        self.spec_augmentor = spec_augmentor

        self.labels = [
            tokenizer.label_from_digits(s)
            for s in self.df["transcription"].astype(str)
        ]

        print(f"  Loading {len(self.audio_paths)} audio files into RAM...")
        self._audio = [load_audio(p, target_sr) for p in tqdm(self.audio_paths, leave=False)]

    def __getitem__(self, idx):
        audio = self._audio[idx]
        if self.waveform_augmentor is not None:
            audio = self.waveform_augmentor(audio.copy())
        features = self._extract(audio)
        if self.spec_augmentor is not None:
            features = self.spec_augmentor(features)
        row = self.df.iloc[idx]
        return {
            "features": features,
            "feature_length": features.shape[0],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "label_length": len(self.labels[idx]),
            "filename": str(row["filename"]),
            "transcription": str(row["transcription"]),
            "spk_id": str(row["spk_id"]),
        }

    def collate_fn(self, batch):
        out = super().collate_fn(batch)
        label_lens = [b["label_length"] for b in batch]
        max_label_len = max(label_lens)
        padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
        for i, b in enumerate(batch):
            padded_labels[i, : b["label_length"]] = b["labels"]
        out["labels"] = padded_labels
        out["label_lengths"] = torch.tensor(label_lens, dtype=torch.long)
        out["transcriptions"] = [b["transcription"] for b in batch]
        out["spk_ids"] = [b["spk_id"] for b in batch]
        return out


def create_dataloaders(cfg):
    """Build train and validation dataloaders from a Config object."""
    from .augmentation import WaveformAugmentor, SpecAugment

    tokenizer = build_tokenizer(cfg.data.tokenizer)

    waveform_aug = spec_aug = None
    if cfg.aug.enabled:
        waveform_aug = WaveformAugmentor(sample_rate=cfg.data.target_sr)
        spec_aug = SpecAugment(
            freq_mask_param=cfg.aug.freq_mask_param,
            time_mask_param=cfg.aug.time_mask_param,
            n_freq_masks=cfg.aug.n_freq_masks,
            n_time_masks=cfg.aug.n_time_masks,
        )

    train_ds = RussianSpeechDataset(
        data_root=cfg.data.data_root, csv_path=None, tokenizer=tokenizer,
        audio_subdir="train", target_sr=cfg.data.target_sr, n_mels=cfg.data.n_mels,
        hop_length=cfg.data.hop_length, n_fft=cfg.data.n_fft, win_length=cfg.data.win_length,
        waveform_augmentor=waveform_aug, spec_augmentor=spec_aug,
    )
    dev_ds = RussianSpeechDataset(
        data_root=cfg.data.data_root, csv_path=None, tokenizer=tokenizer,
        audio_subdir="dev", target_sr=cfg.data.target_sr, n_mels=cfg.data.n_mels,
        hop_length=cfg.data.hop_length, n_fft=cfg.data.n_fft, win_length=cfg.data.win_length,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              collate_fn=train_ds.collate_fn, num_workers=0, pin_memory=False)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            collate_fn=dev_ds.collate_fn, num_workers=0, pin_memory=False)
    return train_loader, dev_loader, tokenizer


def create_test_dataloader(cfg, batch_size: int = 32):
    ds = BaseSpeechDataset(
        data_root=cfg.data.data_root, csv_path=None, audio_subdir="test",
        target_sr=cfg.data.target_sr, n_mels=cfg.data.n_mels,
        hop_length=cfg.data.hop_length, n_fft=cfg.data.n_fft, win_length=cfg.data.win_length,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=ds.collate_fn, num_workers=0, pin_memory=False)
