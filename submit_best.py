"""Submit using best Optuna trial params.

Usage:
    python submit_best.py --ckpt logs/trial_N/best_model.pth
    python submit_best.py --ckpt logs/trial_N/best_model.pth --out submission.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from best_config import best_config
from data_processor.data import BaseSpeechDataset, load_audio
from data_processor.postprocessor import NormalizedRussianToDigit
from main import load_model


def submit(cfg, ckpt_path: Path, out_path: Path):
    model, tokenizer, device = load_model(cfg, ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params | vocab={len(tokenizer)} | device={device}")

    ds = BaseSpeechDataset(
        data_root=cfg.data.data_root, csv_path=None, audio_subdir="test",
        target_sr=cfg.data.target_sr, n_mels=cfg.data.n_mels,
        hop_length=cfg.data.hop_length, n_fft=cfg.data.n_fft, win_length=cfg.data.win_length,
    )

    to_digits = NormalizedRussianToDigit()
    rows = []

    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc="Inference"):
            audio = load_audio(ds.audio_paths[idx], cfg.data.target_sr)
            filename = str(ds.df.iloc[idx]["filename"])

            features = ds._extract(audio).unsqueeze(0).to(device)
            enc_len = model.get_encoder_lengths(
                torch.tensor([features.shape[1]], dtype=torch.long)
            )
            log_probs = model.get_log_probs(features)
            decoded = model.decode(log_probs, enc_len)
            rows.append((filename, to_digits.convert(tokenizer.join(decoded[0]))))

    pd.DataFrame(rows, columns=["filename", "transcription"]).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to best_model.pth")
    parser.add_argument("--out",  type=Path, default=Path("submission.csv"))
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    cfg = best_config()
    if args.data_root:
        cfg.data.data_root = args.data_root

    submit(cfg, args.ckpt, args.out)
