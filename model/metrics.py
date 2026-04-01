import jiwer
import torch

def calculate_wer(predictions, references):
    """Calculate Word Error Rate"""
    return jiwer.wer(references, predictions)

def calculate_cer(predictions, references):
    """Calculate Character Error Rate (better for numbers)"""
    return jiwer.cer(references, predictions)


def eval_metrics(model, loader, device, tokenizer):

    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in loader:
            audio = batch['audio'].to(device)
            audio_lengths = batch['audio_lengths']
            
            log_probs = model.get_log_probs(audio)
            
            # Greedy decoding
            predicted_ids = torch.argmax(log_probs, dim=-1)
            
            # Decode each sample
            for i in range(len(predicted_ids)):
                pred_tokens = predicted_ids[i][:audio_lengths[i]]
                pred_text = tokenizer.decode(pred_tokens.cpu().numpy())
                predictions.append(pred_text)
                
                ref_text = tokenizer.decode(batch['target'][i][:batch['target_lengths'][i]].cpu().numpy())
                references.append(ref_text)

    wer = calculate_wer(predictions, references)
    cer = calculate_cer(predictions, references)

    print(f"Test WER: {wer:.2f}%")
    print(f"Test CER: {cer:.2f}%")