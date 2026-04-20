"""Hyperparameter search with Optuna. Each trial is a fully isolated training run."""
import optuna

from config import Config
from train import train_model


def objective(trial: optuna.Trial) -> float:
    cfg = Config()

    cfg.model.d_model     = trial.suggest_categorical("d_model", [64, 128, 256])
    cfg.model.nhead       = trial.suggest_categorical("nhead", [2, 4, 8])
    cfg.model.num_layers  = trial.suggest_int("num_layers", 4, 12)
    cfg.model.kernel_size = trial.suggest_categorical("kernel_size", [9, 15, 31])
    cfg.model.dropout     = trial.suggest_float("dropout", 0.05, 0.3)

    cfg.train.lr          = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.train.batch_size  = trial.suggest_categorical("batch_size", [32, 64])

    cfg.aug.freq_mask_param = trial.suggest_int("freq_mask", 10, 40)
    cfg.aug.time_mask_param = trial.suggest_int("time_mask", 40, 100)

    # cfg.decoder.type      = trial.suggest_categorical("decoder", ["greedy", "constrained"])
    cfg.decoder.type      = trial.suggest_categorical("decoder", ["constrained"])

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
    # n_jobs=1 for single GPU; set CUDA_VISIBLE_DEVICES per trial for multi-GPU
    study.optimize(objective, n_trials=50, n_jobs=1)

    print("\nBest trial:")
    t = study.best_trial
    print(f"  score: {t.value:.2f}%")
    for k, v in t.params.items():
        print(f"  {k}: {v}")
