import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_processor.postprocessor import RussianNumberTokenizer
from data_processor.data import RussianNumberDataset
import torch
from model.encoder import ConformerCTC
from train import train_model

def main():
    # Load your dataset
    # Assuming CSV with columns: 'audio_path', 'transcript'
    df = pd.read_csv('dataset.csv')
    
    # Split into train/val
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Create tokenizer from training transcripts
    tokenizer = RussianNumberTokenizer(transcripts=train_df['transcript'].tolist())
    
    # Create datasets
    train_dataset = RussianNumberDataset(
        audio_paths=train_df['audio_path'].tolist(),
        transcripts=train_df['transcript'].tolist(),
        tokenizer=tokenizer
    )
    
    val_dataset = RussianNumberDataset(
        audio_paths=val_df['audio_path'].tolist(),
        transcripts=val_df['transcript'].tolist(),
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=4
    )
    
    # Initialize Conformer model (from your previous implementation)
    model = ConformerCTC(
        input_dim=80,  # n_mels
        vocab_size=len(tokenizer),
        d_model=256,
        nhead=4,
        num_layers=16
    )
    
    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == "__main__":
    main()