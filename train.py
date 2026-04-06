import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import csv
from tqdm import tqdm
from pathlib import Path

def train_model(model, train_loader, val_loader, tokenizer,
                epochs=100, device='cuda', lr=1e-3,
                log_dir='logs', save_best=True, patience=10):
    """
    Training loop with CTC loss and extensive logging.

    Args:
        log_dir: directory to save logs (TensorBoard, CSV, best model)
        save_best: whether to save best model checkpoint
        patience: early stopping patience (0 = no early stopping)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Setup logging directories
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path / 'tensorboard')
    csv_file = log_path / 'loss_log.csv'

    # Initialize CSV log
    with open(csv_file, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(1, epochs + 1):
        # -------------------- Training --------------------
        model.train()
        total_train_loss = 0
        num_batches = len(train_loader)

        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]')
        for step, batch in enumerate(pbar):
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths']
            labels = batch['labels'].to(device)
            label_lengths = batch['label_lengths']
            encoder_lengths = torch.ceil(feature_lengths.float() / 4).long()

            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            loss = F.ctc_loss(
                log_probs, labels, encoder_lengths, label_lengths,
                blank=tokenizer.pad_id, reduction='mean'
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            # Log per-batch loss every 50 steps to TensorBoard
            if step % 50 == 0:
                writer.add_scalar('Loss/train_batch', loss.item(),
                                  (epoch-1)*num_batches + step)

        avg_train_loss = total_train_loss / num_batches
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # -------------------- Validation --------------------
        model.eval()
        total_val_loss = 0
        val_steps = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f'Epoch {epoch}/{epochs} [Val]')
            for batch in pbar_val:
                features = batch['features'].to(device)
                feature_lengths = batch['feature_lengths']
                encoder_lengths = torch.ceil(feature_lengths.float() / 4).long()
                labels = batch['labels'].to(device)
                label_lengths = batch['label_lengths']

                logits = model(features)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

                loss = F.ctc_loss(
                    log_probs, labels, encoder_lengths, label_lengths,
                    blank=tokenizer.pad_id, reduction='mean'
                )

                total_val_loss += loss.item()
                val_steps += 1
                pbar_val.set_postfix({'val_loss': loss.item()})

        avg_val_loss = total_val_loss / val_steps
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)

        # Log to CSV
        with open(csv_file, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, avg_train_loss, avg_val_loss, current_lr])

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # Print summary
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        # -------------------- Model checkpointing & early stopping --------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            if save_best:
                torch.save(model.state_dict(), log_path / 'best_model.pth')
                print(f"  -> New best model saved (val_loss = {avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

    writer.close()
    print(f"\nTraining finished. Best validation loss: {best_val_loss:.4f}")
    return history