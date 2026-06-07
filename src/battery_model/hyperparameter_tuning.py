"""Optuna hyperparameter tuning for the Battery SAC model."""

import logging
import os
from datetime import datetime

import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.battery_model.battery_env import RandomWindowEnv


def setup_optuna_logging(log_dir):
    """Setup logging for Optuna optimization process."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"optuna_optimization_{timestamp}.log")

    logger = logging.getLogger("optuna_logger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)

    return logger, log_file


def _make_tuning_env(prices_full, price_scale):
    """Create the training environment used during Optuna trials."""
    env = RandomWindowEnv(prices_full, window_len=24, price_scale=price_scale)
    return Monitor(env)


def _optuna_objective(
    trial,
    prices_full,
    price_scale,
    log_dir,
    total_timesteps=50000,
    logger=None,
):
    """Optuna objective for SAC hyperparameter tuning."""
    trial_log_dir = os.path.join(log_dir, f"optuna_trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    try:
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
        update_ratio = trial.suggest_categorical("update_ratio", [0.25, 0.5, 1.0, 2.0])
        gradient_steps = max(1, int(train_freq * update_ratio))

        model_kwargs = {
            "buffer_size": trial.suggest_categorical("buffer_size", [100000, 200000, 500000]),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "tau": trial.suggest_float("tau", 0.001, 0.05, log=True),
            "gamma": trial.suggest_float("gamma", 0.995, 0.99999),
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "ent_coef": trial.suggest_categorical("ent_coef", ["auto", "auto_0.1", 0.003, 0.01, 0.03]),
            "learning_starts": trial.suggest_int("learning_starts", 1000, 10000),
            "tensorboard_log": trial_log_dir,
            "verbose": 0,
        }

        if logger:
            logger.info(f"Starting trial {trial.number} with params: {model_kwargs}")

        env_train = DummyVecEnv([lambda: _make_tuning_env(prices_full, price_scale)])
        model = SAC("MlpPolicy", env_train, **model_kwargs)
        model.learn(total_timesteps=total_timesteps)

        mean_reward, _ = evaluate_policy(model, env_train, n_eval_episodes=100, deterministic=True)
        env_train.close()

        if logger:
            logger.info(f"Trial {trial.number} completed with mean reward: {mean_reward:.4f}")

        return mean_reward

    except Exception as e:
        if logger:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}", exc_info=True)
        else:
            print(f"Trial {trial.number} failed with error: {str(e)}")

        return float("-inf")


def tune_hyperparameters(
    prices_full,
    price_scale,
    log_dir="./logs",
    n_trials=50,
    total_timesteps=50000,
):
    """Run Optuna study and return best hyperparameters."""
    logger, log_file = setup_optuna_logging(log_dir)

    logger.info(f"Starting Optuna hyperparameter optimization with {n_trials} trials")
    logger.info(f"Total timesteps per trial: {total_timesteps}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _optuna_objective(
            trial,
            prices_full,
            price_scale,
            log_dir,
            total_timesteps,
            logger,
        ),
        n_trials=n_trials,
    )

    logger.info(f"Best reward: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Optimization completed. Log file: {log_file}")

    print("Best reward:", study.best_value)
    print("Best params:", study.best_params)
    return study.best_params
