import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from config import Config
from data_processor.data import build_tokenizer, create_test_dataloader
from data_processor.postprocessor import RussianToDigitLevenshtein
from best_config import best_config
from train import _build_model, train_model


def load_model(cfg: Config, ckpt_path: Path):
    """Build model + tokenizer, load checkpoint, return (model.eval(), tokenizer, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer(cfg.data.tokenizer)
    model = _build_model(cfg, tokenizer).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model.eval(), tokenizer, device


def submit(cfg: Config, ckpt_path: Path, out_path: Path):
    model, tokenizer, device = load_model(cfg, ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params | vocab={len(tokenizer)} | device={device}")
    loader = create_test_dataloader(cfg)
    to_digits = RussianToDigitLevenshtein()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            features = batch["features"].to(device)
            enc_lens = model.get_encoder_lengths(batch["feature_lengths"])
            log_probs = model.get_log_probs(features)
            decoded = model.decode(log_probs, enc_lens)
            for fn, tokens in zip(batch["filenames"], decoded):
                rows.append((fn, to_digits.convert(tokenizer.join(tokens))))
    pd.DataFrame(rows, columns=["filename", "transcription"]).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Russian ASR for numbers")
    parser.add_argument("-s", "--submit", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=Path("logs/best_model.pth"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--data-root", default=None, help="Override Config.data.data_root")
    parser.add_argument("--tokenizer", default=None, help="char | word | numeric")
    parser.add_argument("--decoder", default=None, help="greedy | beam | constrained")
    args = parser.parse_args()

    cfg = best_config()
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.tokenizer:
        cfg.data.tokenizer = args.tokenizer
    if args.decoder:
        cfg.decoder.type = args.decoder

    if args.submit:
        submit(cfg, args.ckpt, args.out)
    else:
        train_model(cfg)


if __name__ == "__main__":
    main()
