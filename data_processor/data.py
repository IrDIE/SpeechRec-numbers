import os
import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .mel import MelSpectrogramExtractor
from .postprocessor import DigitToRussian, RussianWordTokenizer


class BaseSpeechDataset(Dataset):
    """Audio loading + mel feature extraction. Sufficient for test/inference."""

    def __init__(
        self, data_root, csv_path, audio_subdir, target_sr=16000, n_mels=80
    ):
        self.data_root = Path(data_root)
        self.audio_subdir = audio_subdir
        self.target_sr = target_sr

        if csv_path is None:
            csv_path = self.data_root / f"{audio_subdir}.csv"
        self.df = pd.read_csv(csv_path, sep=",")

        # Resolve audio paths (filenames may already include the subdir prefix).
        self.audio_paths = []
        for filename in self.df["filename"]:
            if "/" in filename or "\\" in filename:
                full_path = self.data_root / filename
            else:
                full_path = self.data_root / self.audio_subdir / filename
            self.audio_paths.append(full_path)

        self.feature_extractor = MelSpectrogramExtractor(
            sample_rate=target_sr, n_mels=n_mels
        )

    def load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(str(path))
            audio = waveform.squeeze().numpy()
        except Exception:
            audio, sr = librosa.load(str(path), sr=None, mono=True)

        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio.astype(np.float32)

    def _extract_features(self, idx):
        audio = self.load_audio(self.audio_paths[idx])
        return self.feature_extractor.extract(audio)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        features = self._extract_features(idx)
        return {
            "features": features,
            "feature_length": features.shape[0],
            "filename": str(self.df.iloc[idx]["filename"]),
        }

    def collate_fn(self, batch):
        feat_lens = [b["feature_length"] for b in batch]
        max_feat_len = max(feat_lens)
        n_mels = batch[0]["features"].shape[1]
        padded_features = torch.zeros(len(batch), max_feat_len, n_mels)
        for i, b in enumerate(batch):
            padded_features[i, : b["feature_length"], :] = b["features"]
        return {
            "features": padded_features,
            "feature_lengths": torch.tensor(feat_lens, dtype=torch.long),
            "filenames": [b["filename"] for b in batch],
        }


class RussianSpeechDataset(BaseSpeechDataset):
    """Adds label encoding, on-disk caching, and speaker metadata for train/val."""

    def __init__(
        self,
        data_root,
        csv_path,
        tokenizer,
        audio_subdir,
        target_sr=16000,
        n_mels=80,
        cache_dir=None,
    ):
        super().__init__(data_root, csv_path, audio_subdir, target_sr, n_mels)
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir) if cache_dir else None

        converter = DigitToRussian()
        self.df["spoken"] = (
            self.df["transcription"].astype(str).apply(converter.convert)
        )

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, idx):
        audio_path = self.audio_paths[idx]
        mtime = os.path.getmtime(audio_path)
        unique_str = f"{audio_path.absolute()}_{mtime}_{self.target_sr}"
        return hashlib.sha256(unique_str.encode()).hexdigest()

    def _build_item(self, idx):
        features = self._extract_features(idx)
        labels = self.tokenizer.encode(self.df.iloc[idx]["spoken"])
        row = self.df.iloc[idx]
        return {
            "features": features,
            "feature_length": features.shape[0],
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_length": len(labels),
            "filename": str(row["filename"]),
            "transcription": str(row["transcription"]),
            "spk_id": str(row["spk_id"]),
        }

    def __getitem__(self, idx):
        if not self.cache_dir:
            return self._build_item(idx)
        cache_file = self.cache_dir / f"{self._get_cache_key(idx)}.pt"
        if cache_file.exists():
            return torch.load(cache_file)
        item = self._build_item(idx)
        torch.save(item, cache_file)
        return item

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


def create_dataloaders(
    data_root_train,
    data_root_dev,
    batch_size=32,
    num_workers=4,
    target_sr=16000,
    n_mels=80,
    train_cache=None,
    dev_cache=None,
):
    """Build train and validation dataloaders using Russian words as targets."""
    # ---- Step 1: Build tokenizer vocabulary from training data ----
    train_csv_path = Path(data_root_train) / "train.csv"
    df_train = pd.read_csv(train_csv_path, sep=",")
    converter = DigitToRussian()
    all_words = set()
    for digit_str in df_train["transcription"].astype(str):
        all_words.update(converter.convert(digit_str).split())
    tokenizer = RussianWordTokenizer(word_vocab=all_words)

    # ---- Step 2: Create datasets ----
    train_dataset = RussianSpeechDataset(
        data_root=data_root_train,
        csv_path=None,
        tokenizer=tokenizer,
        audio_subdir="train",
        target_sr=target_sr,
        n_mels=n_mels,
        cache_dir=train_cache,
    )
    dev_dataset = RussianSpeechDataset(
        data_root=data_root_dev,
        csv_path=None,
        tokenizer=tokenizer,
        audio_subdir="dev",
        target_sr=target_sr,
        n_mels=n_mels,
        cache_dir=dev_cache,
    )

    # ---- Step 3: Dataloaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, dev_loader, tokenizer


def create_test_dataloader(
    data_root, batch_size=32, num_workers=4, target_sr=16000, n_mels=80
):
    """Build a test dataloader (no labels, no speaker info)."""
    dataset = BaseSpeechDataset(
        data_root=data_root,
        csv_path=None,
        audio_subdir="test",
        target_sr=target_sr,
        n_mels=n_mels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
