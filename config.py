from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 8
    kernel_size: int = 9
    dropout: float = 0.1
    num_subsample: int = 3        # 2^n total downsampling; 3 → 8× (≈21 frames/2sec)
    ff_expansion: int = 4


@dataclass
class DataConfig:
    data_root: str = "data"
    n_mels: int = 80
    target_sr: int = 16000
    hop_length: int = 200        # mel frame stride in samples (200 @ 16kHz = 12.5ms)
    tokenizer: str = "numeric"   # "char" | "word" | "numeric"


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
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
