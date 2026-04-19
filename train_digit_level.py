"""
Digit-level classification training.
Uses NumberModel (model/digitbased.py): 6 independent digit heads over a
mean-pooled Conformer encoder. Each head predicts one digit position (0-9),
supporting numbers 0–999 999.
"""
import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_processor.augmentation import build_train_augmentation
from data_processor.mel import MelSpectrogramExtractor
from eval import compute_score
from model.digitbased import NumberModel

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data"
# DATA_ROOT = "/mnt/d/ITMO/2026-SpeechRec/gp1/data/"
CKPT_PATH = Path("logs_digit/best_model.pth")

# audio
N_MELS    = 80
TARGET_SR = 16_000

# model
N_DIGITS    = 6     # supports 0 – 999 999
D_MODEL     = 192
NUM_LAYERS  = 4
NUM_HEADS   = 4
FFN_DIM     = 384   # 2× d_model
KERNEL_SIZE = 31
DROPOUT     = 0.1
AUX_LENGTH  = False  # aux head: predict number of significant digits

# training
BATCH_SIZE      = 64
LR              = 1e-4
EPOCHS          = 100
PATIENCE        = 10          # early-stopping patience
GRAD_CLIP       = float("inf") # 1.0
LR_FACTOR       = 0.99999  #0.5       # ReduceLROnPlateau factor
LR_PATIENCE     = 100_000000#5          # ReduceLROnPlateau patience
# ─────────────────────────────────────────────────────────────────────────────


def _pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_label(transcription: str) -> list[int]:
    """Zero-pad number string to N_DIGITS digits."""
    return [int(d) for d in str(int(transcription)).zfill(N_DIGITS)]


def _preds_to_strings(logits: torch.Tensor) -> list[str]:
    """logits [B, N_DIGITS, 10] → list of stripped digit strings."""
    rows = logits.argmax(dim=-1).cpu().tolist()  # [B, N_DIGITS]
    return [str(int("".join(map(str, row)))) for row in rows]


# ── dataset ───────────────────────────────────────────────────────────────────

class DigitSpeechDataset(Dataset):
    def __init__(self, data_root: Path, split: str, augment: bool = False):
        data_root = Path(data_root)
        self.extractor = MelSpectrogramExtractor(sample_rate=TARGET_SR, n_mels=N_MELS)  # uses config constants

        df = pd.read_csv(data_root / f"{split}.csv")
        self.filenames      = list(df["filename"])
        self.transcriptions = [str(t) for t in df["transcription"]]
        self.labels         = [_to_label(t) for t in self.transcriptions]
        self.spk_ids        = list(df["spk_id"].astype(str)) if "spk_id" in df else [""] * len(df)

        self.audio_paths = [
            data_root / (fn if ("/" in fn or "\\" in fn) else f"{split}/{fn}")
            for fn in self.filenames
        ]

        print(f"  Loading {len(self.audio_paths)} audio files into RAM...")
        self._audio = [self._load(p) for p in tqdm(self.audio_paths, leave=False)]

        self.waveform_aug = self.spec_aug = None
        if augment:
            self.waveform_aug, self.spec_aug = build_train_augmentation(TARGET_SR)

    @staticmethod
    def _load(path) -> np.ndarray:
        try:
            wav, sr = torchaudio.load(str(path))
            audio = wav.squeeze().numpy()
        except Exception:
            audio, sr = librosa.load(str(path), sr=None, mono=True)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        m = np.abs(audio).max()
        return (audio / m if m > 0 else audio).astype(np.float32)

    def __len__(self):
        return len(self._audio)

    def __getitem__(self, idx):
        audio = self._audio[idx].copy()
        if self.waveform_aug:
            audio = self.waveform_aug(audio)
        feats = self.extractor.extract(audio)
        if self.spec_aug:
            feats = self.spec_aug(feats)
        return {
            "features":      feats,
            "feat_len":      feats.shape[0],
            "labels":        torch.tensor(self.labels[idx], dtype=torch.long),
            "transcription": self.transcriptions[idx],
            "spk_id":        self.spk_ids[idx],
            "filename":      self.filenames[idx],
        }

    @staticmethod
    def collate_fn(batch):
        feat_lens = [b["feat_len"] for b in batch]
        padded = torch.zeros(len(batch), max(feat_lens), batch[0]["features"].shape[1])
        for i, b in enumerate(batch):
            padded[i, : b["feat_len"]] = b["features"]
        return {
            "features":       padded,
            "feat_lens":      torch.tensor(feat_lens, dtype=torch.long),
            "labels":         torch.stack([b["labels"] for b in batch]),  # [B, N_DIGITS]
            "transcriptions": [b["transcription"] for b in batch],
            "spk_ids":        [b["spk_id"] for b in batch],
            "filenames":      [b["filename"] for b in batch],
        }


def _forward(model, feats, lengths, labels):
    """Run forward + loss, unpacking aux head if present."""
    out  = model(feats, lengths)
    logits = out[0] if isinstance(out, tuple) else out
    loss = F.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1))
    if isinstance(out, tuple):                         # aux length head
        n_sig = (labels > 0).int().sum(dim=1).clamp(max=model.n_digits) - 1  # 0-indexed class
        loss = loss + 0.1 * F.cross_entropy(out[1], n_sig)
    return logits, loss


def _make_loader(split: str, augment: bool = False) -> DataLoader:
    ds = DigitSpeechDataset(DATA_ROOT, split, augment=augment)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"),
                      collate_fn=DigitSpeechDataset.collate_fn, num_workers=0)


# ── training ──────────────────────────────────────────────────────────────────

def train():
    train_loader = _make_loader("train", augment=True)
    val_loader   = _make_loader("dev")

    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    ind_speakers = set(train_loader.dataset.spk_ids)
    device = _pick_device()

    model = NumberModel(input_dim=N_MELS, d_model=D_MODEL, num_layers=NUM_LAYERS,
                       num_heads=NUM_HEADS, ffn_dim=FFN_DIM, kernel_size=KERNEL_SIZE,
                       dropout=DROPOUT, aux_length=AUX_LENGTH).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LR_FACTOR, patience=LR_PATIENCE)

    log_path = Path(CKPT_PATH.parent)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path / "tensorboard")

    with open(log_path / "loss_log.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_score = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        total_loss = 0
        train_preds, train_refs, train_spks = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for step, batch in enumerate(pbar, 1):
            feats   = batch["features"].to(device)
            lengths = batch["feat_lens"].to(device)
            labels  = batch["labels"].to(device)          # [B, N_DIGITS]

            logits, loss = _forward(model, feats, lengths, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / step:.4f}"})
            train_preds.extend(_preds_to_strings(logits.detach()))
            train_refs.extend(batch["transcriptions"])
            train_spks.extend(batch["spk_ids"])

        avg_train = total_loss / len(train_loader)

        # ── val ──
        model.eval()
        total_val = 0
        val_preds, val_refs, val_spks = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                feats   = batch["features"].to(device)
                lengths = batch["feat_lens"].to(device)
                labels  = batch["labels"].to(device)

                logits, val_loss = _forward(model, feats, lengths, labels)
                total_val += val_loss.item()
                val_preds.extend(_preds_to_strings(logits))
                val_refs.extend(batch["transcriptions"])
                val_spks.extend(batch["spk_ids"])

        avg_val = total_val / len(val_loader)
        scheduler.step(avg_val)
        lr = optimizer.param_groups[0]["lr"]

        train_score = compute_score(train_preds, train_refs, [s in ind_speakers for s in train_spks])
        val_score   = compute_score(val_preds,   val_refs,   [s in ind_speakers for s in val_spks])

        writer.add_scalars("Loss",  {"train": avg_train, "val": avg_val}, epoch)
        writer.add_scalars("Score", {"train": train_score["ind_cer"], "val": val_score["score"]}, epoch)
        writer.add_scalars("Val_CER", {"ind": val_score["ind_cer"], "ood": val_score["ood_cer"]}, epoch)
        writer.add_scalar("LR", lr, epoch)

        with open(log_path / "loss_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train, avg_val, lr])

        print(f"\nEpoch {epoch}  train={avg_train:.4f}  val={avg_val:.4f}  lr={lr:.6f}")
        print(f"  Train CER: {train_score['ind_cer']:.2f}%")
        print(f"  Val   CER: ind={val_score['ind_cer']:.2f}%  ood={val_score['ood_cer']:.2f}%  score={val_score['score']:.2f}%")
        print(f"  Sample: pred={train_preds[0]}  ref={train_refs[0]}")

        if val_score["score"] < best_score:
            best_score = val_score["score"]
            no_improve = 0
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  -> Best model saved (score={best_score:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs")
                break

    writer.close()
    print(f"\nDone. Best val score: {best_score:.2f}%")


# ── inference / submit ────────────────────────────────────────────────────────

def submit(ckpt_path: Path, out_path: Path):
    from data_processor.data import create_test_dataloader

    device = _pick_device()
    model = NumberModel(input_dim=N_MELS, d_model=D_MODEL, num_layers=NUM_LAYERS,
                       num_heads=NUM_HEADS, ffn_dim=FFN_DIM, kernel_size=KERNEL_SIZE,
                       dropout=DROPOUT, aux_length=AUX_LENGTH).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    loader = create_test_dataloader(data_root=DATA_ROOT, batch_size=32, num_workers=0)
    rows: list[tuple[str, str]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            feats   = batch["features"].to(device)
            lengths = batch["feat_lens"] if "feat_lens" in batch else batch["feature_lengths"]
            logits  = model(feats, lengths.to(device))
            for filename, pred in zip(batch["filenames"], _preds_to_strings(logits)):
                rows.append((filename, pred))

    pd.DataFrame(rows, columns=["filename", "transcription"]).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    parser.add_argument("--out",  type=Path, default=Path("submission_digit.csv"))
    args = parser.parse_args()
    submit(args.ckpt, args.out) if args.submit else train()


if __name__ == "__main__":
    main()
