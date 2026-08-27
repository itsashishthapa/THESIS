"""
Training script for Cookie Factory (SteamEnv) using SAC or TD3.
"""

# Run with all parameters:
# python -m src.cookie_factory.train --algorithm TD3 --data cookie_model/WKA_Pe_merged.csv --model-path src/cookie_factory/cookie_td3_binary_beta --log-dir logs/cookie_factory/td3_binary_beta --timesteps 20000 --window-len 24 --forecast-h 24 --price-col Pe --wind-col WKA --seed 42

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.cookie_factory.cookie_factory_env import RandomWindowEnv

DEFAULT_DATA_PATH = "cookie_model/WKA_Pe_merged.csv"
DEFAULT_MODEL_PATH = "src/cookie_factory/cookie_sac_model"
DEFAULT_ALGORITHM = "SAC"
DEFAULT_MODEL_PATHS = {
    "SAC": DEFAULT_MODEL_PATH,
    # ``cookie_td3_model`` was trained with the retired 27-feature observation
    # layout.  Keep new TD3 checkpoints separate from that incompatible artifact.
    "TD3": "src/cookie_factory/cookie_td3_binary_beta",
}
MODEL_CLASSES = {
    "SAC": SAC,
    "TD3": TD3,
}


def normalize_algorithm(algorithm: str) -> str:
    """Return a supported Stable-Baselines3 continuous-control algorithm."""
    algorithm = algorithm.upper()
    if algorithm not in MODEL_CLASSES:
        choices = ", ".join(MODEL_CLASSES)
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Choose one of: {choices}"
        )
    return algorithm


def get_model_class(algorithm: str = DEFAULT_ALGORITHM):
    """Return the Stable-Baselines3 implementation for the requested algorithm."""
    return MODEL_CLASSES[normalize_algorithm(algorithm)]


def get_model_path(algorithm: str = DEFAULT_ALGORITHM) -> str:
    """Return the default saved-model path for SAC or TD3."""
    return DEFAULT_MODEL_PATHS[normalize_algorithm(algorithm)]


def linear_schedule(initial_value: float):
    """Linear learning-rate schedule that decays to 10% of the initial value."""

    def func(progress_remaining: float) -> float:
        return float(np.max([progress_remaining * initial_value, initial_value * 1e-1]))

    return func


def load_cookie_data(
    data_path: str = DEFAULT_DATA_PATH,
    price_col: str = "Pe",
    wind_col: str = "WKA",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load (price, wind) trajectories from CSV or Excel dataset.

    If a CSV is provided the function will attempt to auto-detect the
    separator. The function returns the full series arrays and scaling
    values used by the env.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset: {data_path}")

    # Support CSV and Excel. Allow pandas to infer CSV separator.
    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path, sep=None, engine="python")
    else:
        df = pd.read_excel(data_path)

    if price_col not in df.columns or wind_col not in df.columns:
        raise ValueError(f"Dataset must contain columns '{price_col}' and '{wind_col}'")

    # coerce to numeric and fill/leave NaNs for downstream decisions
    g_grid_full = (
        pd.to_numeric(df[price_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    P_WT_full = (
        pd.to_numeric(df[wind_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )

    price_scale = float(np.percentile(np.abs(g_grid_full), 95) + 1e-6)
    wind_scale = float(np.percentile(np.abs(P_WT_full), 95) + 1e-6)

    print(f"Loaded {len(g_grid_full)} timesteps from {data_path}")
    print(f"price_scale (95th pct): {price_scale:.4f}")
    print(f"wind_scale  (95th pct): {wind_scale:.4f}")

    return g_grid_full, P_WT_full, price_scale, wind_scale


def make_train_env(
    g_grid_full: np.ndarray,
    P_WT_full: np.ndarray,
    window_len: int,
    forecast_h: int,
    price_scale: float,
    wind_scale: float,
    enforce_cyclic_boundary: bool = True,
    env_kwargs: dict[str, Any] | None = None,
):
    """Create a monitored random-window training environment."""
    kwargs = {
        "forecast_h": forecast_h,
        "price_scale": price_scale,
        "wind_scale": wind_scale,
        "enforce_cyclic_boundary": enforce_cyclic_boundary,
    }
    if env_kwargs:
        kwargs.update(env_kwargs)

    env = RandomWindowEnv(
        g_grid_full=g_grid_full,
        P_WT_full=P_WT_full,
        window_len=window_len,
        **kwargs,
    )
    return Monitor(env)


def train_model(
    g_grid_full: np.ndarray,
    P_WT_full: np.ndarray,
    price_scale: float,
    wind_scale: float,
    model_path: str | None = None,
    log_dir: str = "logs/cookie_factory",
    total_timesteps: int = 200_000,
    window_len: int = 24,
    forecast_h: int = 24,
    enforce_cyclic_boundary: bool = True,
    seed: int = 42,
    model_kwargs: dict[str, Any] | None = None,
    algorithm: str = DEFAULT_ALGORITHM,
) -> SAC | TD3:
    """Train SAC or TD3 and save the final training checkpoint."""
    algorithm = normalize_algorithm(algorithm)
    model_class = get_model_class(algorithm)
    model_path = model_path or get_model_path(algorithm)
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    env = DummyVecEnv(
        [
            lambda: make_train_env(
                g_grid_full=g_grid_full,
                P_WT_full=P_WT_full,
                window_len=window_len,
                forecast_h=forecast_h,
                price_scale=price_scale,
                wind_scale=wind_scale,
                enforce_cyclic_boundary=enforce_cyclic_boundary,
            )
        ]
    )

    model_init_kwargs: dict[str, Any] = {
        "buffer_size": 200_000,
        "batch_size": 256,
        "learning_rate": linear_schedule(3e-4),
        "tau": 0.005,
        "gamma": 1,
        "train_freq": 1,
        "gradient_steps": 1,
        "verbose": 1,
        "tensorboard_log": log_dir,
        "seed": seed,
    }
    # if algorithm == "SAC":
    #     model_init_kwargs["policy_kwargs"] = {"net_arch": [256, 256, 256], "activation_fn": nn.ReLU}
    if algorithm == "TD3":
        # model_init_kwargs["policy_kwargs"] = {"net_arch": [256, 256, 256], "activation_fn": nn.ReLU}
        n_actions = env.action_space.shape[-1]
        model_init_kwargs.update(
            {
                "action_noise": NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=0.2 * np.ones(n_actions),
                ),
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
                "learning_starts": 1_000,
            }
        )
    if model_kwargs:
        model_init_kwargs.update(model_kwargs)

    # Normalize observations and rewards to stabilize training
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        gamma=model_init_kwargs.get("gamma", 0.99),
    )

    model = model_class("MlpPolicy", env, **model_init_kwargs)
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )
    model.save(model_path)
    env.save(f"{model_path}_vecnormalize.pkl")

    inference_env = DummyVecEnv(
        [
            lambda: make_train_env(
                g_grid_full=g_grid_full,
                P_WT_full=P_WT_full,
                window_len=window_len,
                forecast_h=forecast_h,
                price_scale=price_scale,
                wind_scale=wind_scale,
                enforce_cyclic_boundary=enforce_cyclic_boundary,
            )
        ]
    )
    inference_env = VecNormalize.load(
        f"{model_path}_vecnormalize.pkl", inference_env
    )
    inference_env.training = False
    inference_env.norm_reward = False
    model = model_class.load(model_path, env=inference_env)

    print(f"{algorithm} training complete. Final model saved to {model_path}.zip")
    print(f"Matching VecNormalize stats saved to {model_path}_vecnormalize.pkl")

    env.close()
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC or TD3 on Cookie Factory SteamEnv"
    )
    parser.add_argument(
        "--algorithm",
        type=normalize_algorithm,
        default=DEFAULT_ALGORITHM,
        choices=("SAC", "TD3"),
        help="RL algorithm to train",
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA_PATH, help="Path to CSV or Excel dataset"
    )
    parser.add_argument("--model-path", default=None, help="Where to save the model")
    parser.add_argument("--log-dir", default=None, help="TensorBoard/log directory")
    parser.add_argument(
        "--timesteps", type=int, default=20_000, help="Total training steps"
    )
    parser.add_argument(
        "--window-len", type=int, default=24, help="Random training window length"
    )
    parser.add_argument(
        "--forecast-h", type=int, default=24, help="Forecast horizon in env"
    )
    parser.add_argument(
        "--price-col",
        type=str,
        default="Pe",
        help="Column name for grid price (CSV/Excel)",
    )
    parser.add_argument(
        "--wind-col",
        type=str,
        default="WKA",
        help="Column name for wind power (CSV/Excel)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    g_grid_full, P_WT_full, price_scale, wind_scale = load_cookie_data(
        data_path=args.data,
        price_col=args.price_col,
        wind_col=args.wind_col,
    )
    if len(g_grid_full) < args.window_len:
        raise ValueError(
            f"Training data must contain at least {args.window_len} timesteps."
        )
    print(f"Training data: {len(g_grid_full)} timesteps")

    model_path = args.model_path or get_model_path(args.algorithm)
    log_dir = (
        args.log_dir or f"logs/cookie_factory/{args.algorithm.lower()}_binary_beta"
    )
    train_model(
        g_grid_full=g_grid_full,
        P_WT_full=P_WT_full,
        price_scale=price_scale,
        wind_scale=wind_scale,
        model_path=model_path,
        log_dir=log_dir,
        total_timesteps=args.timesteps,
        window_len=args.window_len,
        forecast_h=args.forecast_h,
        enforce_cyclic_boundary=not args.no_cyclic_boundary,
        seed=args.seed,
        algorithm=args.algorithm,
    )


if __name__ == "__main__":
    main()
