import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from data_processor.data import create_dataloaders, create_test_dataloader
from data_processor.postprocessor import RussianCharTokenizer, RussianToDigit, RussianWordTokenizer
from model.decoder import GreedyDecoder
from model.encoder import ConformerCTC
from train import train_model


DATA_ROOT = "data"
TOKENIZER = "char"  # "word" or "char"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_model(tokenizer) -> ConformerCTC:
    return ConformerCTC(
        input_dim=80,
        vocab_size=len(tokenizer),
        d_model=256,
        nhead=4,
        num_layers=8,
        decoder=GreedyDecoder(tokenizer),
    )


def train():
    train_loader, val_loader, tokenizer = create_dataloaders(
        data_root_train=DATA_ROOT,
        data_root_dev=DATA_ROOT,
        batch_size=64,
        num_workers=0,
        target_sr=16000,
        n_mels=80,
        train_cache="cache/train",
        dev_cache="cache/dev",
        tokenizer_type=TOKENIZER,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Dev samples: {len(val_loader.dataset)}")
    print(f"Vocabulary size: {len(tokenizer)}")

    sample_batch = next(iter(train_loader))
    print(f"Features shape: {sample_batch['features'].shape}")

    model = _build_model(tokenizer)
    n_params_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    print(f"Model parameters: {n_params_M:.2f}M")

    ind_speakers = set(train_loader.dataset.df["spk_id"].astype(str))

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        ind_speakers=ind_speakers,
        epochs=100,
        device=_pick_device(),
        lr=0.0001,
    )


def submit(ckpt_path: Path, out_path: Path):
    device = _pick_device()
    print(f"Using device: {device}")

    tokenizer = RussianCharTokenizer() if TOKENIZER == "char" else RussianWordTokenizer()
    model = _build_model(tokenizer)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    loader = create_test_dataloader(data_root=DATA_ROOT, batch_size=32, num_workers=0)

    to_digits = RussianToDigit()
    rows: list[tuple[str, str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"]
            encoder_lengths = torch.ceil(feature_lengths.float() / 4).long()

            log_probs = model.get_log_probs(features)
            decoded = model.decode(log_probs, encoder_lengths)

            for filename, tokens in zip(batch["filenames"], decoded):
                rows.append((filename, to_digits.convert(tokenizer.join(tokens))))

    pd.DataFrame(rows, columns=["filename", "transcription"]).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit", action="store_true",
                        help="Run inference on test.csv and write submission.csv instead of training.")
    parser.add_argument("--ckpt", type=Path, default=Path("logs/best_model.pth"),
                        help="Checkpoint path for --submit mode.")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"),
                        help="Output CSV path for --submit mode.")
    args = parser.parse_args()

    if args.submit:
        submit(args.ckpt, args.out)
    else:
        train()


if __name__ == "__main__":
    main()
