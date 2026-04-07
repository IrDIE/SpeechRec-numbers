import torch

from data_processor.data import create_dataloaders
from model.decoder import GreedyDecoder
from model.encoder import ConformerCTC
from train import train_model


def main():
    # Paths (adjust to your actual mount points)
    TRAIN_ROOT = "data"
    DEV_ROOT = "data"

    # Create dataloaders
    train_loader, val_loader, tokenizer = create_dataloaders(
        data_root_train=TRAIN_ROOT,
        data_root_dev=DEV_ROOT,
        batch_size=32,
        num_workers=4,
        target_sr=16000,
        n_mels=80,
        train_cache="cache/train",
        dev_cache="cache/dev",
    )

    # Print dataset sizes
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Dev samples: {len(val_loader.dataset)}")
    print(f"Vocabulary size: {len(tokenizer)}")

    # Example: inspect a batch
    sample_batch = next(iter(train_loader))
    print(f"Features shape: {sample_batch['features'].shape}")  # (batch, time, 80)

    # Initialize Conformer model (from your previous implementation)
    model = ConformerCTC(
        input_dim=80,  # n_mels
        vocab_size=len(tokenizer),
        d_model=256,
        nhead=4,
        num_layers=16,
        decoder=GreedyDecoder(tokenizer),
    )
    n_params_M = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    )
    print(f"Model parameters: {n_params_M:.2f}M")

    # In-domain speakers = those seen in training. Used to split val CER.
    ind_speakers = set(train_loader.dataset.df["spk_id"].astype(str))

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        ind_speakers=ind_speakers,
        epochs=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    main()
