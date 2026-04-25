"""Best Optuna trial config (score=0.83%). Single source of truth."""
from config import Config


def best_config() -> Config:
    cfg = Config()
    cfg.data.n_mels        = 80
    cfg.data.hop_length    = 160
    cfg.data.tokenizer     = "normalized_char"
    cfg.data.__post_init__()

    cfg.model.d_model      = 128
    cfg.model.nhead        = 4
    cfg.model.num_layers   = 10
    cfg.model.kernel_size  = 15
    cfg.model.ff_expansion = 4
    cfg.model.dropout      = 0.008033891308982799
    cfg.model.num_subsample = 2

    cfg.train.lr           = 0.0006854678486846137
    cfg.train.weight_decay = 0.0018828754733551862
    cfg.train.betas        = (0.95, 0.999)

    cfg.aug.freq_mask_param = 18
    cfg.aug.time_mask_param = 66
    cfg.aug.n_freq_masks    = 2
    cfg.aug.n_time_masks    = 2

    cfg.decoder.type       = "constrained"
    cfg.decoder.beam_size  = 500
    return cfg
