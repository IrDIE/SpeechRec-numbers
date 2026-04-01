import torch.nn as nn
import torch.nn.functional as F
import torch

def train_model(model, train_loader, val_loader, tokenizer, epochs=100, device='cuda'):
    """
    Training loop with CTC loss.
    """
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths']
            labels = batch['labels'].to(device)
            label_lengths = batch['label_lengths']
            
            # Forward pass
            logits = model(features)  # Shape: (batch, time, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # CTC expects (time, batch, vocab)
            
            # Compute CTC loss
            loss = F.ctc_loss(
                log_probs,
                labels,
                feature_lengths,
                label_lengths,
                blank=tokenizer.pad_id,
                reduction='mean'
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                feature_lengths = batch['feature_lengths']
                labels = batch['labels'].to(device)
                label_lengths = batch['label_lengths']
                
                logits = model(features)
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.transpose(0, 1)
                
                loss = F.ctc_loss(
                    log_probs,
                    labels,
                    feature_lengths,
                    label_lengths,
                    blank=tokenizer.pad_id,
                    reduction='mean'
                )
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch+1}: New best model! Val loss: {avg_val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")