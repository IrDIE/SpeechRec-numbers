# from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
# from transformers import PreTrainedTokenizerFast
from .mel import MelSpectrogramExtractor, AudioPreprocessor

from torch.utils.data import Dataset
import torch

class RussianNumberDataset(Dataset):
    """
    Dataset for Russian number recognition with manual feature extraction.
    """
    
    def __init__(self, audio_paths, transcripts, tokenizer, 
                 target_sr=16000, n_mels=80):
        """
        Args:
            audio_paths: list of audio file paths
            transcripts: list of transcript strings (spoken form)
            tokenizer: RussianNumberTokenizer instance
            target_sr: target sample rate (16kHz)
            n_mels: number of Mel filterbanks
        """
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.tokenizer = tokenizer
        
        # Initialize feature extractor
        self.feature_extractor = MelSpectrogramExtractor(
            sample_rate=target_sr,
            n_mels=n_mels
        )
        
        self.audio_preprocessor = AudioPreprocessor(target_sr=target_sr)
        
        # Precompute features and labels for faster training (optional)
        self.cached_features = []
        self.cached_labels = []
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess audio
        audio, sr = self.audio_preprocessor.load_and_preprocess(
            self.audio_paths[idx]
        )
        
        # Extract Mel spectrogram
        features = self.feature_extractor.extract(audio)  # Shape: (time, n_mels)
        
        # Encode transcript
        transcript = self.transcripts[idx]
        labels = self.tokenizer.encode(transcript)
        
        return {
            'features': features,
            'feature_length': features.shape[0],
            'labels': torch.tensor(labels, dtype=torch.long),
            'label_length': len(labels)
        }
    
    def collate_fn(self, batch):
        """
        Custom collate function for padding variable-length sequences.
        """
        features = []
        feature_lengths = []
        labels = []
        label_lengths = []
        
        for item in batch:
            features.append(item['features'])
            feature_lengths.append(item['feature_length'])
            labels.append(item['labels'])
            label_lengths.append(item['label_length'])
        
        # Pad features
        max_feature_len = max(feature_lengths)
        padded_features = torch.zeros(len(batch), max_feature_len, features[0].shape[1])
        for i, feat in enumerate(features):
            padded_features[i, :feat.shape[0], :] = feat
        
        # Pad labels
        max_label_len = max(label_lengths)
        padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label
        
        return {
            'features': padded_features,
            'feature_lengths': torch.tensor(feature_lengths, dtype=torch.long),
            'labels': padded_labels,
            'label_lengths': torch.tensor(label_lengths, dtype=torch.long)
        }
    

# class RussianNumberProcessor:
#     """Custom processor for Russian number recognition"""
    
#     def __init__(self, model_name="facebook/wav2vec2-base-960h"):
#         # Use standard feature extractor from Wav2Vec2
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
#         # Create custom tokenizer for Russian numbers
#         self.tokenizer = RussianNumberTokenizer()
        
#         # For compatibility with Wav2Vec2Processor API
#         self.processor = Wav2Vec2Processor(
#             feature_extractor=self.feature_extractor,
#             tokenizer=self._create_hf_tokenizer()
#         )
        
#         self.normalizer = RussianNumberNormalizer()
    
#     def _create_hf_tokenizer(self):
#         """Create HuggingFace-compatible tokenizer"""
#         return PreTrainedTokenizerFast(
#             tokenizer_object=None,
#             vocab=self.tokenizer.vocab,
#             pad_token="<pad>",
#             unk_token="<unk>",
#             bos_token="<s>",
#             eos_token="</s>"
#         )
    
#     def extract_features(self, audio_array, sampling_rate=16000):
#         """Extract Mel spectrogram features"""
#         inputs = self.feature_extractor(
#             audio_array,
#             sampling_rate=sampling_rate,
#             return_tensors="pt"
#         )
#         return inputs.input_values
    
#     def encode_text(self, text):
#         """Encode text to token IDs"""
#         return self.tokenizer.encode(text)
    
#     def decode_tokens(self, token_ids):
#         """Decode token IDs to text"""
#         return self.tokenizer.decode(token_ids)
    
#     def normalize_number(self, text):
#         """Normalize Russian numbers to digits"""
#         return self.normalizer.normalize(text)