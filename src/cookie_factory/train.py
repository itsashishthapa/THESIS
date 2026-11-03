"""
Training script for Cookie Factory (SteamEnv) using SAC or TD3.
"""

# Run with all parameters:
# python -m src.cookie_factory.train --algorithm TD3 --data cookie_model/WKA_Pe_merged.csv --model-path src/cookie_factory/cookie_td3_derived --log-dir logs/cookie_factory/td3_derived --timesteps 20000 --window-len 24 --forecast-h 24 --price-col Pe --wind-col WKA --seed 42

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.cookie_factory.cookie_factory_env import RandomWindowEnv, SteamEnv

DEFAULT_DATA_PATH = "cookie_model/WKA_Pe_merged.csv"
DEFAULT_MODEL_PATH = "src/cookie_factory/cookie_SAC_terminal"
DEFAULT_ALGORITHM = "SAC"
DEFAULT_MODEL_PATHS = {
    "SAC": DEFAULT_MODEL_PATH,
    "TD3": "src/cookie_factory/cookie_TD3_terminal",
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


def load_environment_config(model_path: str) -> dict[str, Any]:
    """Read saved input scaling for a derived-R checkpoint."""
    path = f"{model_path}_config.json"
    if not os.path.exists(path):
        return {}
    with open(path) as config_file:
        config = json.load(config_file)
    return {key: config[key] for key in (
        "price_scale", "wind_scale", "forecast_h", "enforce_cyclic_boundary"
    ) if key in config}


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


def split_train_validation_data(
    g_grid_full: np.ndarray,
    P_WT_full: np.ndarray,
    validation_fraction: float,
    window_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return chronological training and validation blocks."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    split_index = int(len(g_grid_full) * (1.0 - validation_fraction))
    g_grid_train, g_grid_validation = g_grid_full[:split_index], g_grid_full[split_index:]
    P_WT_train, P_WT_validation = P_WT_full[:split_index], P_WT_full[split_index:]
    if len(g_grid_train) < window_len or len(g_grid_validation) < window_len:
        raise ValueError(
            "Training and validation blocks must each contain at least "
            f"{window_len} timesteps."
        )
    return g_grid_train, P_WT_train, g_grid_validation, P_WT_validation


class LowestViolationCheckpointCallback(BaseCallback):
    """Save the policy whose validation windows have the lowest violation."""

    def __init__(
        self,
        g_grid_validation: np.ndarray,
        P_WT_validation: np.ndarray,
        price_scale: float,
        wind_scale: float,
        model_path: str,
        window_len: int,
        forecast_h: int,
        eval_freq: int,
        n_validation_windows: int,
        seed: int,
        enforce_cyclic_boundary: bool,
    ):
        super().__init__()
        self.g_grid_validation = np.asarray(g_grid_validation, dtype=np.float32)
        self.P_WT_validation = np.asarray(P_WT_validation, dtype=np.float32)
        self.price_scale = float(price_scale)
        self.wind_scale = float(wind_scale)
        self.model_path = model_path
        self.window_len = int(window_len)
        self.forecast_h = int(forecast_h)
        self.eval_freq = max(int(eval_freq), 1)
        self.enforce_cyclic_boundary = bool(enforce_cyclic_boundary)
        self.best_mean_violation = float("inf")
        self.last_eval_step: int | None = None

        if n_validation_windows <= 0:
            raise ValueError("n_validation_windows must be positive.")
        n_available = len(self.g_grid_validation) - self.window_len + 1
        if n_available <= 0:
            raise ValueError("Validation data is shorter than the evaluation window.")
        rng = np.random.default_rng(seed)
        self.validation_starts = np.sort(
            rng.choice(
                n_available,
                size=min(int(n_validation_windows), n_available),
                replace=False,
            )
        )

    def _window_violation(self, start: int) -> float:
        end = start + self.window_len
        env = SteamEnv(
            self.g_grid_validation[start:end],
            self.P_WT_validation[start:end],
            forecast_h=self.forecast_h,
            price_scale=self.price_scale,
            wind_scale=self.wind_scale,
            enforce_cyclic_boundary=self.enforce_cyclic_boundary,
        )
        observation, _ = env.reset()
        total = 0.0
        done = False
        temperature_range = env.T_s_max - env.T_s_min
        while not done:
            normalized_observation = self.training_env.normalize_obs(observation)
            action, _ = self.model.predict(normalized_observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            total += float(info["bypass_sum_violation_k"])
            total += float(info["complementarity_violation_k"]) / env.Q_flow_max**2
            total += float(info["flow_bound_violation_k"]) / env.Q_flow_max
            total += float(info["T_I_bound_violation_k"]) / (250.0 - 177.0)
            total += float(info["state_violation_k"]) / temperature_range
            total += float(info["grid_bound_violation_k"]) / env.P_HP_max
            total += abs(float(info["f4_residual_k"])) / temperature_range
            done = bool(terminated or truncated)

        if self.enforce_cyclic_boundary:
            total += abs(env.T_s_k - env.T_s_0) / temperature_range
        return total

    def _evaluate(self) -> None:
        mean_violation = float(
            np.mean([self._window_violation(int(start)) for start in self.validation_starts])
        )
        self.logger.record("validation/mean_normalized_violation", mean_violation)
        if mean_violation < self.best_mean_violation:
            self.best_mean_violation = mean_violation
            self.model.save(self.model_path)
            self.training_env.save(f"{self.model_path}_vecnormalize.pkl")
            print(
                f"Validation step {self.num_timesteps}: new best normalized "
                f"violation={mean_violation:.6f}"
            )
        self.last_eval_step = self.num_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self.last_eval_step != self.num_timesteps:
            self._evaluate()


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
    g_grid_validation: np.ndarray | None = None,
    P_WT_validation: np.ndarray | None = None,
    eval_freq: int = 25_000,
    n_validation_windows: int = 100,
    validation_seed: int = 42,
    algorithm: str = DEFAULT_ALGORITHM,
) -> SAC | TD3:
    """Train SAC or TD3 and retain the lowest-violation validation checkpoint."""
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
    )

    model = model_class("MlpPolicy", env, **model_init_kwargs)
    if g_grid_validation is None or P_WT_validation is None:
        raise ValueError("Validation price and wind arrays are required.")
    validation_callback = LowestViolationCheckpointCallback(
        g_grid_validation=g_grid_validation,
        P_WT_validation=P_WT_validation,
        price_scale=price_scale,
        wind_scale=wind_scale,
        model_path=model_path,
        window_len=window_len,
        forecast_h=forecast_h,
        eval_freq=eval_freq,
        n_validation_windows=n_validation_windows,
        seed=validation_seed,
        enforce_cyclic_boundary=enforce_cyclic_boundary,
    )
    with open(f"{model_path}_config.json", "w") as config_file:
        json.dump({"price_scale": price_scale, "wind_scale": wind_scale,
                   "forecast_h": forecast_h,
                   "enforce_cyclic_boundary": enforce_cyclic_boundary}, config_file, indent=2)
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=validation_callback,
    )

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

    print(
        f"{algorithm} training complete. Lowest-violation model saved to "
        f"{model_path}.zip (validation normalized violation: "
        f"{validation_callback.best_mean_violation:.6f})"
    )
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
        "--timesteps", type=int, default=200_000, help="Total training steps"
    )
    parser.add_argument(
        "--window-len", type=int, default=24, help="Random training window length"
    )
    parser.add_argument(
        "--forecast-h", type=int, default=24, help="Forecast horizon in env"
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Chronological fraction reserved for validation",
    )
    parser.add_argument(
        "--validation-windows",
        type=int,
        default=100,
        help="Fixed validation windows evaluated per checkpoint",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Training steps between validation evaluations",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=42,
        help="Seed used to choose fixed validation windows",
    )
    parser.add_argument(
        "--no-cyclic-boundary",
        action="store_true",
        help="Disable the cyclic terminal boundary (F14) penalty",
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
    g_grid_train, P_WT_train, g_grid_validation, P_WT_validation = (
        split_train_validation_data(
            g_grid_full,
            P_WT_full,
            validation_fraction=args.validation_fraction,
            window_len=args.window_len,
        )
    )
    print(
        f"Training data: {len(g_grid_train)} timesteps | "
        f"Validation data: {len(g_grid_validation)} timesteps"
    )

    # Fit fixed input scales on training data only.
    price_scale = float(np.percentile(np.abs(g_grid_train), 95) + 1e-6)
    wind_scale = float(np.percentile(np.abs(P_WT_train), 95) + 1e-6)
    model_path = args.model_path or get_model_path(args.algorithm)
    log_dir = (
        args.log_dir or f"logs/cookie_factory/{args.algorithm.lower()}_derived"
    )
    train_model(
        g_grid_full=g_grid_train,
        P_WT_full=P_WT_train,
        price_scale=price_scale,
        wind_scale=wind_scale,
        model_path=model_path,
        log_dir=log_dir,
        total_timesteps=args.timesteps,
        window_len=args.window_len,
        forecast_h=args.forecast_h,
        enforce_cyclic_boundary=not args.no_cyclic_boundary,
        seed=args.seed,
        g_grid_validation=g_grid_validation,
        P_WT_validation=P_WT_validation,
        eval_freq=args.eval_freq,
        n_validation_windows=args.validation_windows,
        validation_seed=args.validation_seed,
        algorithm=args.algorithm,
    )


if __name__ == "__main__":
    main()
