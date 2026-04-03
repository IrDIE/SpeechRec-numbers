import os
import pandas as pd
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
from .mel import MelSpectrogramExtractor
from .postprocessor import RussianWordTokenizer, DigitToRussian
import torchaudio
import librosa, hashlib
import numpy as np

class RussianSpeechDataset(Dataset):
    def __init__(
        self, data_root, csv_path, tokenizer, audio_subdir, target_sr=16000, n_mels=80, cache_dir=None
    ):
        """
        Args:
            data_root: root directory containing the CSV and audio subfolder
            csv_path: full path to CSV file (if None, uses {data_root}/{audio_subdir}.csv)
            tokenizer: RussianWordTokenizer instance
            audio_subdir: subdirectory name (e.g., "train" or "dev")
            target_sr: target sample rate (16kHz)
            n_mels: number of Mel filterbanks
        """
        self.data_root = Path(data_root)
        self.audio_subdir = audio_subdir
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Load CSV
        if csv_path is None:
            csv_path = self.data_root / f"{audio_subdir}.csv"
        self.df = pd.read_csv(csv_path, sep=",")  # adjust separator if needed

        # Build full audio paths
        self.audio_paths = []
        for filename in self.df["filename"]:
            if "/" in filename or "\\" in filename:
                full_path = self.data_root / filename
            else:
                full_path = self.data_root / self.audio_subdir / filename
            self.audio_paths.append(full_path)

        # Convert digit transcriptions to Russian words (spoken form)
        converter = DigitToRussian()
        self.df["spoken"] = (
            self.df["transcription"].astype(str).apply(converter.convert)
        )

        # Feature extractor (from scratch, no pretrained)
        self.feature_extractor = MelSpectrogramExtractor(
            sample_rate=target_sr, n_mels=n_mels
        )
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(str(path))
            audio = waveform.squeeze().numpy()
        except Exception:
            audio, sr = librosa.load(str(path), sr=None, mono=True)
        
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio.astype(np.float32)

    def __len__(self):
        return len(self.audio_paths)

    def _get_cache_key(self, idx):
        """Generate a unique cache key based on audio file path + modification time."""
        audio_path = self.audio_paths[idx]
        # Use absolute path and modification time to detect changes
        mtime = os.path.getmtime(audio_path)
        unique_str = f"{audio_path.absolute()}_{mtime}_{self.target_sr}"
        hash_obj = hashlib.sha256(unique_str.encode())
        return hash_obj.hexdigest()

    def _get_cached_item(self, idx):
        """Load cached item from disk if exists, else compute and cache."""
        cache_key = self._get_cache_key(idx)
        cache_file = self.cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            # Load cached data
            cached = torch.load(cache_file)
            return cached
        # Compute features and labels
        audio = self.load_audio(self.audio_paths[idx])
        features = self.feature_extractor.extract(audio)  # (T, n_mels)
        target_text = self.df.iloc[idx]["spoken"]
        labels = self.tokenizer.encode(target_text)
        item = {
            'features': features,
            'feature_length': features.shape[0],
            'labels': torch.tensor(labels, dtype=torch.long),
            'label_length': len(labels)
        }
        # Save to cache
        torch.save(item, cache_file)
        return item

    def __getitem__(self, idx):
        if self.cache_dir:
            return self._get_cached_item(idx)
        else:
            # No caching, compute each time
            audio = self.load_audio(self.audio_paths[idx])
            features = self.feature_extractor.extract(audio)
            target_text = self.df.iloc[idx]["spoken"]
            labels = self.tokenizer.encode(target_text)
            return {
                'features': features,
                'feature_length': features.shape[0],
                'labels': torch.tensor(labels, dtype=torch.long),
                'label_length': len(labels)
            }

    def collate_fn(self, batch):
        features = [b["features"] for b in batch]
        feat_lens = [b["feature_length"] for b in batch]
        labels = [b["labels"] for b in batch]
        label_lens = [b["label_length"] for b in batch]

        max_feat_len = max(feat_lens)
        n_mels = features[0].shape[1]
        padded_features = torch.zeros(len(batch), max_feat_len, n_mels)
        for i, f in enumerate(features):
            padded_features[i, : f.shape[0], :] = f

        max_label_len = max(label_lens)
        padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
        for i, lbl in enumerate(labels):
            padded_labels[i, : len(lbl)] = lbl

        return {
            "features": padded_features,
            "feature_lengths": torch.tensor(feat_lens, dtype=torch.long),
            "labels": padded_labels,
            "label_lengths": torch.tensor(label_lens, dtype=torch.long),
        }


def create_dataloaders(
    data_root_train,
    data_root_dev,
    batch_size=32,
    num_workers=4,
    target_sr=16000,
    n_mels=80, train_cache = None, dev_cache = None
):
    """
    Build train and validation dataloaders using Russian words as targets.
    Returns:
        train_loader, dev_loader, tokenizer (built from training vocabulary)
    """
    # ---- Step 1: Build tokenizer vocabulary from training data ----
    train_csv_path = Path(data_root_train) / "train.csv"
    df_train = pd.read_csv(train_csv_path, sep=",")
    converter = DigitToRussian()
    # Collect all unique words from the spoken forms
    all_words = set()
    for digit_str in df_train["transcription"].astype(str):
        spoken = converter.convert(digit_str) # '992597'
        all_words.update(spoken.split())
    # Create tokenizer with the collected vocabulary
    tokenizer = RussianWordTokenizer(word_vocab=all_words)

    # ---- Step 2: Create datasets (they will use the same tokenizer) ----
    train_dataset = RussianSpeechDataset(
        data_root=data_root_train,
        csv_path=None,  # will use data_root_train/train.csv
        tokenizer=tokenizer,
        audio_subdir="train",
        target_sr=target_sr,
        n_mels=n_mels,
        cache_dir=train_cache
    )

    dev_dataset = RussianSpeechDataset(
        data_root=data_root_dev,
        csv_path=None,  # will use data_root_dev/dev.csv
        tokenizer=tokenizer,
        audio_subdir="dev",
        target_sr=target_sr,
        n_mels=n_mels,
        cache_dir=dev_cache
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
