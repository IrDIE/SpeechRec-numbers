"""Hyperparameter search with Optuna. Each trial is a fully isolated training run."""
import optuna

from config import Config
from train import train_model


def _count_params(cfg: Config) -> int:
    """Count model parameters using the tokenizer and architecture from cfg."""
    from data_processor.data import build_tokenizer
    from model.encoder import ConformerCTC
    tokenizer = build_tokenizer(cfg.data.tokenizer)
    m = ConformerCTC(
        input_dim=cfg.data.n_mels,
        vocab_size=len(tokenizer),
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        kernel_size=cfg.model.kernel_size,
        dropout=0.0,
        num_subsample=cfg.model.num_subsample,
    )
    return sum(p.numel() for p in m.parameters())


def objective(trial: optuna.Trial) -> float:
    cfg = Config()

    # Mel features
    cfg.data.n_mels        = trial.suggest_categorical("n_mels", [64, 80, 128])
    cfg.data.hop_length    = trial.suggest_categorical("hop_length", [160, 200, 256])
    cfg.data.__post_init__()  # recompute n_fft and win_length from new hop_length

    # Model architecture
    cfg.model.d_model      = trial.suggest_categorical("d_model", [64, 128, 256])
    cfg.model.nhead        = trial.suggest_categorical("nhead", [2, 4, 8])
    cfg.model.num_layers   = trial.suggest_int("num_layers", 4, 20)
    cfg.model.kernel_size  = trial.suggest_categorical("kernel_size", [9, 15, 31])
    cfg.model.ff_expansion = trial.suggest_categorical("ff_expansion", [2, 4])
    cfg.model.num_subsample = 2
    cfg.model.dropout      = trial.suggest_float("dropout", 0.0, 0.4)

    # Training
    cfg.train.lr           = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    beta1                  = trial.suggest_categorical("beta1", [0.85, 0.9, 0.95])
    cfg.train.betas        = (beta1, 0.999)
    cfg.train.batch_size   = 32 #trial.suggest_categorical("batch_size", [16, 32, 64])

    # Augmentation
    cfg.aug.freq_mask_param = trial.suggest_int("freq_mask", 10, 40)
    cfg.aug.time_mask_param = trial.suggest_int("time_mask", 40, 100)
    cfg.aug.n_freq_masks   = trial.suggest_int("n_freq_masks", 1, 3)
    cfg.aug.n_time_masks   = trial.suggest_int("n_time_masks", 1, 3)

    cfg.decoder.type       = "constrained"

    # Enforce 4M–5M parameter budget
    n_params = _count_params(cfg)
    trial.set_user_attr("n_params_M", round(n_params / 1e6, 2))
    if not (4_000_000 <= n_params <= 5_000_000):
        raise optuna.TrialPruned()

    # Each trial writes to its own directory — fully isolated for parallel runs
    cfg.train.log_dir = f"logs/trial_{trial.number}"

    return train_model(cfg)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="conformer-ctc",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=200, n_jobs=1)

    print("\nBest trial:")
    t = study.best_trial
    print(f"  score: {t.value:.2f}%")
    for k, v in t.params.items():
        print(f"  {k}: {v}")
