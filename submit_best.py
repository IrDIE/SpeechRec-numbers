"""Submit using best Optuna trial params (score=1.37%) with TTA.

TTA: run each audio at multiple speeds (via resampling), average log-probs,
then run one constrained beam decode pass.

Usage:
    python submit_best.py --ckpt logs/trial_N/best_model.pth
    python submit_best.py --ckpt logs/trial_N/best_model.pth --out submission.csv --no-tta
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from best_config import best_config
from data_processor.data import BaseSpeechDataset, build_tokenizer, load_audio
from data_processor.postprocessor import NormalizedRussianToDigit
from main import load_model

TTA_SPEEDS = [0.85, 0.925, 1.0, 1.075, 1.15]  # speed factors via resampling


def resample_audio(audio: np.ndarray, speed: float, sr: int) -> np.ndarray:
    """Simulate speed change by resampling: play audio as if recorded at sr*speed."""
    if speed == 1.0:
        return audio
    orig_freq = int(sr * speed)
    wav = torch.from_numpy(audio).unsqueeze(0)
    resampled = torchaudio.functional.resample(wav, orig_freq=orig_freq, new_freq=sr)
    return resampled.squeeze(0).numpy()


def submit_tta(cfg: Config, ckpt_path: Path, out_path: Path, speeds: list[float]):
    model, tokenizer, device = load_model(cfg, ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params | vocab={len(tokenizer)} | device={device}")
    print(f"TTA speeds: {speeds}")

    ds = BaseSpeechDataset(
        data_root=cfg.data.data_root, csv_path=None, audio_subdir="test",
        target_sr=cfg.data.target_sr, n_mels=cfg.data.n_mels,
        hop_length=cfg.data.hop_length, n_fft=cfg.data.n_fft, win_length=cfg.data.win_length,
    )

    to_digits = NormalizedRussianToDigit()
    rows = []

    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc="Inference (TTA)"):
            audio = load_audio(ds.audio_paths[idx], cfg.data.target_sr)
            filename = str(ds.df.iloc[idx]["filename"])

            avg_log_probs = None
            for speed in speeds:
                aug_audio = resample_audio(audio, speed, cfg.data.target_sr)
                features = ds._extract(aug_audio).unsqueeze(0).to(device)  # (1, T, n_mels)
                feat_len = torch.tensor([features.shape[1]], dtype=torch.long)
                enc_len = model.get_encoder_lengths(feat_len)

                log_probs = model.get_log_probs(features)  # (1, T', vocab)

                if avg_log_probs is None:
                    avg_log_probs = log_probs
                else:
                    # Align lengths (different speeds → different T'); pad shorter to longer
                    t1, t2 = avg_log_probs.shape[1], log_probs.shape[1]
                    if t1 < t2:
                        avg_log_probs = torch.nn.functional.pad(avg_log_probs, (0, 0, 0, t2 - t1))
                        enc_len_avg = enc_len
                    else:
                        log_probs = torch.nn.functional.pad(log_probs, (0, 0, 0, t1 - t2))
                        enc_len_avg = torch.tensor([avg_log_probs.shape[1]], dtype=torch.long)
                    avg_log_probs = avg_log_probs + log_probs

            avg_log_probs = avg_log_probs / len(speeds)
            final_enc_len = model.get_encoder_lengths(
                torch.tensor([avg_log_probs.shape[1]], dtype=torch.long)
            )
            decoded = model.decode(avg_log_probs, final_enc_len)
            rows.append((filename, to_digits.convert(tokenizer.join(decoded[0]))))

    pd.DataFrame(rows, columns=["filename", "transcription"]).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to best_model.pth")
    parser.add_argument("--out",  type=Path, default=Path("submission.csv"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA (speed=1.0 only)")
    args = parser.parse_args()

    cfg = best_config()
    if args.data_root:
        cfg.data.data_root = args.data_root

    speeds = [1.0] if args.no_tta else TTA_SPEEDS
    submit_tta(cfg, args.ckpt, args.out, speeds)
