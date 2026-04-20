import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_processor.data import create_dataloaders
from data_processor.postprocessor import RussianToDigitLevenshtein, NormalizedRussianToDigit
from eval import compute_score
from model.encoder import ConformerCTC


def _build_model(cfg, tokenizer) -> ConformerCTC:
    from model.decoder import ConstrainedBeamDecoder, GreedyDecoder, LMBeamSearchDecoder

    if cfg.decoder.type == "constrained":
        decoder = ConstrainedBeamDecoder(tokenizer, beam_size=cfg.decoder.beam_size)
    elif cfg.decoder.type == "beam":
        decoder = LMBeamSearchDecoder(
            tokenizer, lm_path=cfg.decoder.lm_path,
            beam_size=cfg.decoder.beam_size, lm_weight=cfg.decoder.lm_weight,
        )
    else:
        decoder = GreedyDecoder(tokenizer)

    return ConformerCTC(
        input_dim=cfg.data.n_mels,
        vocab_size=len(tokenizer),
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        decoder=decoder,
        num_subsample=cfg.model.num_subsample,
    )


def train_model(cfg) -> float:
    """Fully isolated training run. Returns best validation score (lower = better).

    Safe for parallel Optuna trials: no global state, writes to cfg.train.log_dir.
    """
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = Path(cfg.train.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tokenizer = create_dataloaders(cfg)
    ind_speakers = set(train_loader.dataset.df["spk_id"].astype(str))

    model = _build_model(cfg, tokenizer).to(device)
    to_digits = (NormalizedRussianToDigit() if cfg.data.tokenizer == "normalized_char"
                 else RussianToDigitLevenshtein())

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params/1e6:.2f}M params | vocab={len(tokenizer)} | device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay, betas=cfg.train.betas)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.train.lr_factor, patience=cfg.train.lr_patience,
    )

    writer = SummaryWriter(log_dir=log_path / "tensorboard")
    csv_file = log_path / "loss_log.csv"
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val_score = float("inf")
    no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        pred_d, ref_d, spk_ids = [], [], []

        for step, batch in enumerate(tqdm(train_loader, desc=f"Ep {epoch} [train]")):
            features = batch["features"].to(device)
            enc_lens = model.get_encoder_lengths(batch["feature_lengths"])
            labels = batch["labels"].to(device)

            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.ctc_loss(
                log_probs.transpose(0, 1), labels, enc_lens, batch["label_lengths"],
                blank=tokenizer.pad_id, reduction="mean", zero_infinity=True,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                decoded = model.decode(log_probs.detach(), enc_lens)
            pred_d.extend(to_digits.convert(tokenizer.join(t)) for t in decoded)
            ref_d.extend(batch["transcriptions"])
            spk_ids.extend(batch["spk_ids"])

            if step % 50 == 0:
                writer.add_scalar("Loss/train_batch", loss.item(),
                                  (epoch - 1) * len(train_loader) + step)

        avg_train = train_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_train, epoch)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_pred, val_ref, val_spk = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch} [val]"):
                features = batch["features"].to(device)
                enc_lens = model.get_encoder_lengths(batch["feature_lengths"])
                labels = batch["labels"].to(device)

                logits = model(features)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = F.ctc_loss(
                    log_probs.transpose(0, 1), labels, enc_lens, batch["label_lengths"],
                    blank=tokenizer.pad_id, reduction="mean", zero_infinity=True,
                )
                val_loss += loss.item()

                decoded = model.decode(log_probs, enc_lens)
                val_pred.extend(to_digits.convert(tokenizer.join(t)) for t in decoded)
                val_ref.extend(batch["transcriptions"])
                val_spk.extend(batch["spk_ids"])

        avg_val = val_loss / len(val_loader)
        writer.add_scalar("Loss/val_epoch", avg_val, epoch)
        scheduler.step(avg_val)
        lr = optimizer.param_groups[0]["lr"]

        train_score = compute_score(pred_d, ref_d, [s in ind_speakers for s in spk_ids])
        val_score = compute_score(val_pred, val_ref, [s in ind_speakers for s in val_spk])

        writer.add_scalars("Score", {"train": train_score["ind_cer"], "val": val_score["score"]}, epoch)
        writer.add_scalars("Val_CER", {"ind": val_score["ind_cer"], "ood": val_score["ood_cer"]}, epoch)

        with open(csv_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train, avg_val, lr])

        print(f"\nEp {epoch}: train={avg_train:.4f} val={avg_val:.4f} lr={lr:.2e}")
        print(f"  train_cer={train_score['ind_cer']:.2f}%")
        print(f"  val: ind={val_score['ind_cer']:.2f}% ood={val_score['ood_cer']:.2f}% score={val_score['score']:.2f}%")
        print(f"  pred={pred_d[0]!r}  ref={ref_d[0]!r}")

        if val_score["score"] < best_val_score:
            best_val_score = val_score["score"]
            no_improve = 0
            torch.save(model.state_dict(), log_path / "best_model.pth")
            print(f"  -> saved (score={best_val_score:.2f}%)")
        else:
            no_improve += 1
            if cfg.train.patience > 0 and no_improve >= cfg.train.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    print(f"Done. Best val score: {best_val_score:.2f}%")
    return best_val_score
