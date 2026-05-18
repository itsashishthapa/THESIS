"""
Standalone inference script for Cookie Factory SAC model.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from stable_baselines3 import SAC

from src.cookie_factory.cookie_factory_env import SteamEnv

DEFAULT_DATA_PATH = "cookie_model/data_24h_test.xlsx"
DEFAULT_MODEL_PATH = "src/cookie_factory/cookie_sac_model"
DEFAULT_INFERENCE_OUTPUT = "results/metrics/cookie_inference_rollout.csv"


def load_cookie_data(
    excel_path: str = DEFAULT_DATA_PATH,
    price_col_idx: int = 2,
    wind_col_idx: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load (price, wind) trajectories from the cookie-model Excel file."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Could not find dataset: {excel_path}")

    df = pd.read_excel(excel_path)
    arr = df.to_numpy()

    if arr.ndim != 2:
        raise ValueError("Expected 2D table data in Excel file")
    if arr.shape[1] <= max(price_col_idx, wind_col_idx):
        raise ValueError(
            f"Excel file has {arr.shape[1]} columns, but requested "
            f"price_col_idx={price_col_idx}, wind_col_idx={wind_col_idx}"
        )

    g_grid_full = arr[:, price_col_idx].astype(float)
    P_WT_full = arr[:, wind_col_idx].astype(float)

    price_scale = float(np.percentile(np.abs(g_grid_full), 95) + 1e-6)
    wind_scale = float(np.percentile(np.abs(P_WT_full), 95) + 1e-6)

    print(f"Loaded {len(g_grid_full)} timesteps from {excel_path}")
    print(f"price_scale (95th pct): {price_scale:.4f}")
    print(f"wind_scale  (95th pct): {wind_scale:.4f}")

    return g_grid_full, P_WT_full, price_scale, wind_scale


def run_inference(
    model: SAC,
    g_grid: np.ndarray,
    P_WT: np.ndarray,
    forecast_h: int,
    price_scale: float,
    wind_scale: float,
    enforce_cyclic_boundary: bool = True,
    seed: int = 42,
    output_csv: str | None = DEFAULT_INFERENCE_OUTPUT,
) -> dict[str, float]:
    """Run deterministic inference on SteamEnv and optionally save rollout CSV."""
    env = SteamEnv(
        g_grid=g_grid,
        P_WT=P_WT,
        forecast_h=forecast_h,
        price_scale=price_scale,
        wind_scale=wind_scale,
        enforce_cyclic_boundary=enforce_cyclic_boundary,
    )

    obs, _ = env.reset(seed=seed)
    done = False
    step_idx = 0
    cumulative_reward = 0.0
    rows: list[dict[str, float]] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        cumulative_reward += float(reward)
        rows.append(
            {
                "k": step_idx,
                "reward": float(reward),
                "cumulative_reward": float(cumulative_reward),
                "beta1_k": float(info.get("beta1_k", np.nan)),
                "beta2_k": float(info.get("beta2_k", np.nan)),
                "R_k": float(info.get("R_k", np.nan)),
                "m_I_k": float(info.get("m_I_k", np.nan)),
                "T_I_k": float(info.get("T_I_k", np.nan)),
                "T_s_k": float(info.get("T_s_k", np.nan)),
                "P_grid_k": float(info.get("P_grid_k", np.nan)),
                "cost_grid_k": float(info.get("cost_grid_k", np.nan)),
                "mode_violation_k": float(info.get("mode_violation_k", np.nan)),
                "state_violation_k": float(info.get("state_violation_k", np.nan)),
                "complementarity_violation_k": float(info.get("complementarity_violation_k", np.nan)),
                "f4_residual_k": float(info.get("f4_residual_k", np.nan)),
                "f5_residual_k": float(info.get("f5_residual_k", np.nan)),
            }
        )
        step_idx += 1

    rollout_df = pd.DataFrame(rows)
    metrics = {
        "steps": float(len(rollout_df)),
        "cumulative_reward": float(rollout_df["reward"].sum()),
        "total_cost": float(rollout_df["cost_grid_k"].sum()),
        "mean_mode_violation": float(rollout_df["mode_violation_k"].mean()),
        "mean_state_violation": float(rollout_df["state_violation_k"].mean()),
        "mean_complementarity_violation": float(rollout_df["complementarity_violation_k"].mean()),
        "mean_abs_f4_residual": float(rollout_df["f4_residual_k"].abs().mean()),
        "mean_abs_f5_residual": float(rollout_df["f5_residual_k"].abs().mean()),
    }

    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        rollout_df.to_csv(output_csv, index=False)
        print(f"Inference rollout saved to {output_csv}")

    print("Inference summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for Cookie Factory SAC model")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to Excel dataset")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to trained model (without .zip)")
    parser.add_argument("--forecast-h", type=int, default=24, help="Forecast horizon")
    parser.add_argument("--no-cyclic-boundary", action="store_true", help="Disable cyclic terminal boundary (F14) penalty in env")
    parser.add_argument("--price-col", type=int, default=2, help="Excel column index for grid price")
    parser.add_argument("--wind-col", type=int, default=1, help="Excel column index for wind power")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=DEFAULT_INFERENCE_OUTPUT, help="CSV output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    g_grid_full, P_WT_full, price_scale, wind_scale = load_cookie_data(
        excel_path=args.data,
        price_col_idx=args.price_col,
        wind_col_idx=args.wind_col,
    )

    model = SAC.load(args.model_path)

    run_inference(
        model=model,
        g_grid=g_grid_full,
        P_WT=P_WT_full,
        forecast_h=args.forecast_h,
        price_scale=price_scale,
        wind_scale=wind_scale,
        enforce_cyclic_boundary=not args.no_cyclic_boundary,
        seed=args.seed,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
