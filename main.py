import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from typing import Literal
from data_processor.data import create_dataloaders, create_test_dataloader
from data_processor.postprocessor import RussianCharTokenizer, RussianWordTokenizer, RussianToDigitLevenshtein
from model.decoder import ConstrainedBeamDecoder, GreedyDecoder, LMBeamSearchDecoder
from model.encoder import ConformerCTC
from train import train_model

# ── config (edit here; notebook imports these) ────────────────────────────────
POSSIBLE_DECODERS = Literal["greedy", "beam", "constrained"]
DATA_ROOT = "data"
# DATA_ROOT = "/mnt/d/ITMO/2026-SpeechRec/gp1/data/"
TOKENIZER = "char"      # "word" or "char"
DECODER   = "greedy"    # "greedy" | "beam" | "constrained"
CKPT_PATH = Path("logs/best_model.pth")
LM_DECODER_PATH = "speechtotext_ru_ru_lm_deployable_v2.0/4gram-pruned-0_1_7_9-ru-lm-set-1.0.bin"
# ─────────────────────────────────────────────────────────────────────────────


def build_tokenizer():
    return RussianCharTokenizer() if TOKENIZER == "char" else RussianWordTokenizer()


def build_model(tokenizer, decoder_type: POSSIBLE_DECODERS = DECODER) -> ConformerCTC:
    if decoder_type == "beam":
        decoder = LMBeamSearchDecoder(tokenizer, lm_path=LM_DECODER_PATH, beam_size=20, lm_weight=0.0, word_score=0.0)
    elif decoder_type == "constrained":
        decoder = ConstrainedBeamDecoder(tokenizer, beam_size=50)
    else:
        decoder = GreedyDecoder(tokenizer)

    return ConformerCTC(
        input_dim=80,
        vocab_size=len(tokenizer),
        d_model=128,
        nhead=4,
        num_layers=8,
        decoder=decoder,
        kernel_size=9,
    )


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train():
    train_loader, val_loader, tokenizer = create_dataloaders(
        data_root_train=DATA_ROOT,
        data_root_dev=DATA_ROOT,
        batch_size=64,
        num_workers=0,
        target_sr=16000,
        n_mels=80,
        tokenizer_type=TOKENIZER,
        augment_train=True,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Dev samples: {len(val_loader.dataset)}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Features shape: {next(iter(train_loader))['features'].shape}")

    model = build_model(tokenizer)
    n_params_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    print(f"Model parameters: {n_params_M:.2f}M")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        ind_speakers=set(train_loader.dataset.df["spk_id"].astype(str)),
        epochs=100,
        device=_pick_device(),
        lr=0.0001,
    )


def submit(ckpt_path: Path, out_path: Path):
    device = _pick_device()
    print(f"Using device: {device}")

    tokenizer = build_tokenizer()
    model = build_model(tokenizer).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    loader = create_test_dataloader(data_root=DATA_ROOT, batch_size=32, num_workers=0)
    to_digits = RussianToDigitLevenshtein()
    rows: list[tuple[str, str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            features = batch["features"].to(device)
            encoder_lengths = model.get_encoder_lengths(batch["feature_lengths"])
            log_probs = model.get_log_probs(features)
            decoded = model.decode(log_probs, encoder_lengths)
            for filename, tokens in zip(batch["filenames"], decoded):
                rows.append((filename, to_digits.convert(tokenizer.join(tokens))))

    pd.DataFrame(rows, columns=["filename", "transcription"]).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    parser.add_argument("--out",  type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    submit(args.ckpt, args.out) if args.submit else train()


if __name__ == "__main__":
    main()
