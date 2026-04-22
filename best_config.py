"""Best Optuna trial config (score=1.37%). Single source of truth."""
from config import Config


def best_config() -> Config:
    cfg = Config()
    cfg.data.n_mels        = 128
    cfg.data.hop_length    = 160
    cfg.data.tokenizer     = "normalized_char"
    cfg.data.__post_init__()

    cfg.model.d_model      = 128
    cfg.model.nhead        = 4
    cfg.model.num_layers   = 9
    cfg.model.kernel_size  = 9
    cfg.model.ff_expansion = 2
    cfg.model.dropout      = 0.37233889704266926
    cfg.model.num_subsample = 2

    cfg.decoder.type       = "constrained"
    cfg.decoder.beam_size  = 500
    return cfg
