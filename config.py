from dataclasses import dataclass, field as dc_field


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 8
    kernel_size: int = 9
    dropout: float = 0.1
    num_subsample: int = 2        # 2^n total downsampling; 2 → 4× (≈40 frames/2sec)
    ff_expansion: int = 4


@dataclass
class DataConfig:
    data_root: str = "data"
    n_mels: int = 80
    target_sr: int = 16000
    hop_length: int = 200        # mel frame stride in samples (200 @ 16kHz = 12.5ms)
    tokenizer: str = "normalized_char"  # "char" | "word" | "numeric" | "normalized_char"

    # Derived from hop_length: win_length = 2 × hop_length, n_fft = win_length
    win_length: int = dc_field(init=False)
    n_fft: int = dc_field(init=False)

    def __post_init__(self):
        self.win_length = self.hop_length * 2
        self.n_fft = self.win_length


@dataclass
class AugConfig:
    enabled: bool = True
    freq_mask_param: int = 27
    time_mask_param: int = 70
    n_freq_masks: int = 2
    n_time_masks: int = 2


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.999)
    patience: int = 10
    grad_clip: float = 1.0
    lr_factor: float = 0.5
    lr_patience: int = 5
    log_dir: str = "logs"
    seed: int = 42


@dataclass
class DecoderConfig:
    type: str = "constrained"  # "greedy" | "beam" | "constrained"
    beam_size: int = 50
    lm_path: str = ""
    lm_weight: float = 0.5


@dataclass
class Config:
    model: ModelConfig = dc_field(default_factory=ModelConfig)
    data: DataConfig = dc_field(default_factory=DataConfig)
    aug: AugConfig = dc_field(default_factory=AugConfig)
    train: TrainConfig = dc_field(default_factory=TrainConfig)
    decoder: DecoderConfig = dc_field(default_factory=DecoderConfig)
