"""Train SAC on the soft-constrained Cookie Factory environment."""

# Run with all parameters:
# python -m src.cookie_factory.train_no_hard_constrains --data cookie_model/WKA_Pe_merged.csv --model-path src/cookie_factory/cookie_sac_model_no_hard_constrains --log-dir logs/cookie_factory_no_hard_constrains --timesteps 20000 --window-len 24 --forecast-h 24 --no-cyclic-boundary --price-col Pe --wind-col WKA --seed 42

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.cookie_factory.cookie_factory_env_no_hard_constrains import RandomWindowEnv

DEFAULT_DATA_PATH = "cookie_model/WKA_Pe_merged.csv"
DEFAULT_MODEL_PATH = "src/cookie_factory/cookie_sac_model_no_hard_constrains"
DEFAULT_LOG_DIR = "logs/cookie_factory_no_hard_constrains"


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
    g_grid_full = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    P_WT_full = pd.to_numeric(df[wind_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

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
    """Create a monitored random-window soft-constrained environment."""
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
    model_path: str = DEFAULT_MODEL_PATH,
    log_dir: str = DEFAULT_LOG_DIR,
    total_timesteps: int = 200_000,
    window_len: int = 24,
    forecast_h: int = 24,
    enforce_cyclic_boundary: bool = True,
    seed: int = 42,
    model_kwargs: dict[str, Any] | None = None,
) -> SAC:
    """Train and save a SAC policy using only soft operational constraints."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

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

    sac_kwargs: dict[str, Any] = {
        "buffer_size": 200_000,
        "batch_size": 256,
        "learning_rate": linear_schedule(3e-4),
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "verbose": 1,
        "tensorboard_log": log_dir,
        "seed": seed,
    }
    if model_kwargs:
        sac_kwargs.update(model_kwargs)

    # Normalize observations and rewards to stabilize training
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=sac_kwargs.get("gamma", 0.99))

    model = SAC("MlpPolicy", env, **sac_kwargs)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)
    env.save(f"{model_path}_vecnormalize.pkl")

    print(f"Training complete. Model saved to {model_path}.zip")
    print(f"VecNormalize stats saved to {model_path}_vecnormalize.pkl")

    env.close()
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on soft-constrained Cookie Factory SteamEnv")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV or Excel dataset")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Where to save the model")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="TensorBoard/log directory")
    parser.add_argument("--timesteps", type=int, default=20_000, help="Total SAC training steps")
    parser.add_argument("--window-len", type=int, default=24, help="Random training window length")
    parser.add_argument("--forecast-h", type=int, default=24, help="Forecast horizon in env")
    parser.add_argument("--no-cyclic-boundary", action="store_true", help="Disable cyclic terminal boundary (F14) penalty in env")
    parser.add_argument("--price-col", type=str, default="Pe", help="Column name for grid price (CSV/Excel)")
    parser.add_argument("--wind-col", type=str, default="WKA", help="Column name for wind power (CSV/Excel)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    g_grid_full, P_WT_full, price_scale, wind_scale = load_cookie_data(
        data_path=args.data,
        price_col=args.price_col,
        wind_col=args.wind_col,
    )

    train_model(
        g_grid_full=g_grid_full,
        P_WT_full=P_WT_full,
        price_scale=price_scale,
        wind_scale=wind_scale,
        model_path=args.model_path,
        log_dir=args.log_dir,
        total_timesteps=args.timesteps,
        window_len=args.window_len,
        forecast_h=args.forecast_h,
        enforce_cyclic_boundary=not args.no_cyclic_boundary,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
