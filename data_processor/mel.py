import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample

class MelSpectrogramExtractor:
    """
    Manual Mel spectrogram extraction from scratch.
    No pretrained components - all parameters initialized fresh.
    """
    
    def __init__(self, 
                 sample_rate=16000,
                 n_mels=80,
                 n_fft=400,      # 25ms at 16kHz: 0.025 * 16000 = 400
                 hop_length=160,  # 10ms at 16kHz: 0.010 * 16000 = 160
                 win_length=400,
                 f_min=0,
                 f_max=8000):
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        
        # Create Mel filter bank (from scratch, no pretrained weights)
        self.mel_filterbank = self._create_mel_filterbank()
        
        # Create Hann window
        self.window = torch.hann_window(win_length)
    
    def _create_mel_filterbank(self):
        """
        Create Mel filterbank from scratch using standard formulas.
        This is deterministic and doesn't use any pretrained weights.
        """
        # Convert Hz to Mel
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Create Mel scale points
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # FFT frequencies
        fft_freqs = np.linspace(0, self.sample_rate // 2, self.n_fft // 2 + 1)
        
        # Create filterbank
        filterbank = np.zeros((self.n_mels, len(fft_freqs)))
        
        for i in range(self.n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            for j, freq in enumerate(fft_freqs):
                if left < freq < center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center < freq < right:
                    filterbank[i, j] = (right - freq) / (right - center)
                elif freq == center:
                    filterbank[i, j] = 1.0
        
        # Normalize filterbank
        filterbank = filterbank / np.sum(filterbank, axis=1, keepdims=True)
        
        return torch.tensor(filterbank, dtype=torch.float32)
    
    def extract(self, audio):
        """
        Extract Mel spectrogram from audio waveform.
        
        Args:
            audio: numpy array or torch tensor of shape (samples,)
        
        Returns:
            mel_spec: torch tensor of shape (time, n_mels)
        """
        # Convert to torch tensor if numpy
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure audio is 1D
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        
        # Compute power spectrogram
        power_spec = stft.abs().pow(2)  # Shape: (freq_bins, time)
        
        # Apply Mel filterbank
        # power_spec shape: (freq_bins, time)
        # mel_filterbank shape: (n_mels, freq_bins)
        mel_spec = torch.matmul(self.mel_filterbank, power_spec)
        
        # Convert to log scale (log10)
        mel_spec = torch.log10(mel_spec + 1e-6)
        
        # Transpose to (time, n_mels) for easier handling
        mel_spec = mel_spec.transpose(0, 1)
        
        return mel_spec
    

class AudioPreprocessor:
    """
    Handle audio loading and resampling to 16kHz from various sample rates.
    """
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
    
    def load_audio(self, audio_path):
        """
        Load audio file and resample to target sample rate.
        Supports various input sample rates (22.05kHz, 44.1kHz, etc.)
        """
        # Load with librosa (automatically handles various formats)
        audio, original_sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample if necessary
        if original_sr != self.target_sr:
            audio = librosa.resample(
                audio, 
                orig_sr=original_sr, 
                target_sr=self.target_sr,
                res_type='kaiser_best'
            )
        
        # Normalize audio to [-1, 1] range
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        
        return audio.astype(np.float32), self.target_sr
    
    def load_and_preprocess(self, audio_path):
        """Complete preprocessing pipeline"""
        audio, sr = self.load_audio(audio_path)
        return audio, sr