"""
Evaluation script for Cookie Factory SAC model.
Runs deterministic (greedy) rollouts and collects metrics.
"""

# Run with all parameters:
# python -m src.cookie_factory.evaluate --model-path src/cookie_factory/cookie_sac_model --data cookie_model/WKA_Pe_merged.csv --num-episodes 10 --window-len 24 --forecast-h 24 --stochastic --seed 42 --output-dir results/eval --price-col Pe --wind-col WKA --first-24

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.cookie_factory.cookie_factory_env import SteamEnv
from src.cookie_factory.train import load_cookie_data

DEFAULT_MODEL_PATH = "src/cookie_factory/cookie_sac_model"
DEFAULT_DATA_PATH = "cookie_model/WKA_Pe_merged.csv"
DEFAULT_VECNORM_PATH = "src/cookie_factory/cookie_sac_model_vecnormalize.pkl"
FIRST_24_ROWS = 24


def load_first_24_cookie_data(
    data_path: str = DEFAULT_DATA_PATH,
    price_col: str = "Pe",
    wind_col: str = "WKA",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load the same first 24 WKA/Pe rows used by cookie_optimization.py."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset: {data_path}")

    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path, sep=None, engine="python", nrows=FIRST_24_ROWS)
    else:
        df = pd.read_excel(data_path, nrows=FIRST_24_ROWS)

    if price_col not in df.columns or wind_col not in df.columns:
        raise ValueError(f"Dataset must contain columns '{price_col}' and '{wind_col}'")
    if len(df) < FIRST_24_ROWS:
        raise ValueError(f"Dataset must contain at least {FIRST_24_ROWS} rows")

    g_grid = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    P_WT = pd.to_numeric(df[wind_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    price_scale = float(np.percentile(np.abs(g_grid), 95) + 1e-6)
    wind_scale = float(np.percentile(np.abs(P_WT), 95) + 1e-6)

    print(f"Loaded first {FIRST_24_ROWS} timesteps from {data_path}")
    print(f"price_scale (95th pct): {price_scale:.4f}")
    print(f"wind_scale  (95th pct): {wind_scale:.4f}")

    return g_grid, P_WT, price_scale, wind_scale


def evaluate_first_24_data(
    model_path: str = DEFAULT_MODEL_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    price_col: str = "Pe",
    wind_col: str = "WKA",
    forecast_h: int = 24,
    deterministic: bool = True,
    seed: int = 42,
    env_kwargs: dict[str, Any] | None = None,
    vecnormalize_path: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate one episode on the first 24 rows of WKA_Pe_merged.csv and print it."""
    if vecnormalize_path is None:
        vecnormalize_path = f"{model_path}_vecnormalize.pkl"

    g_grid, P_WT, price_scale, wind_scale = load_first_24_cookie_data(
        data_path=data_path,
        price_col=price_col,
        wind_col=wind_col,
    )

    episode_stats, trajectories = evaluate_model(
        model_path=model_path,
        g_grid_full=g_grid,
        P_WT_full=P_WT,
        price_scale=price_scale,
        wind_scale=wind_scale,
        num_episodes=1,
        window_len=FIRST_24_ROWS,
        forecast_h=forecast_h,
        deterministic=deterministic,
        seed=seed,
        env_kwargs=env_kwargs,
        vecnormalize_path=vecnormalize_path,
    )

    result = episode_stats[0]
    print("\nFirst 24-row evaluation result:")
    print(pd.DataFrame([result]).to_string(index=False))
    if not trajectories.empty:
        violation_columns = [
            "constraint_violation_bypass_post",
            "constraint_violation_complementarity_post",
            "constraint_violation_state",
            "constraint_violation_grid_post",
            "constraint_violation_coupling",
        ]
        violation_summary = trajectories[violation_columns].sum().sort_values(ascending=False).reset_index()
        violation_summary.columns = ["constraint", "total_violation"]
        violation_summary["mean_violation_per_step"] = violation_summary["total_violation"] / len(trajectories)
        print("\nFirst 24-row violation breakdown:")
        print(violation_summary.to_string(index=False))

        f4_total = trajectories["constraint_violation_f4"].sum()
        f5_total = trajectories["constraint_violation_f5"].sum()
        coupling_detail = pd.DataFrame([
            {
                "constraint": "constraint_violation_f4",
                "total_violation": f4_total,
                "mean_violation_per_step": f4_total / len(trajectories),
            },
            {
                "constraint": "constraint_violation_f5",
                "total_violation": f5_total,
                "mean_violation_per_step": f5_total / len(trajectories),
            },
        ])
        print("\nFirst 24-row coupling detail:")
        print(coupling_detail.to_string(index=False))

        f4_infeasible = trajectories.loc[
            trajectories["f4_projection_feasible"] == False,
            "f4_infeasible_reasons",
        ]
        f4_reason_summary = f4_infeasible.explode().dropna().value_counts().rename_axis("reason").reset_index(name="steps")
        if not f4_reason_summary.empty:
            print("\nFirst 24-row F4 infeasibility reasons:")
            print(f4_reason_summary.to_string(index=False))
        mode_repairs = trajectories.loc[
            trajectories["f4_mode_repair_applied"],
            "f4_mode_candidate",
        ].value_counts().rename_axis("mode").reset_index(name="steps")
        if not mode_repairs.empty:
            print("\nFirst 24-row F4 joint mode repairs:")
            print(mode_repairs.to_string(index=False))

    return result, trajectories


def evaluate_model(
    model_path: str,
    g_grid_full: np.ndarray,
    P_WT_full: np.ndarray,
    price_scale: float,
    wind_scale: float,
    num_episodes: int = 10,
    window_len: int = 24,
    forecast_h: int = 24,
    deterministic: bool = True,
    seed: int = 42,
    env_kwargs: dict[str, Any] | None = None,
    vecnormalize_path: str | None = DEFAULT_VECNORM_PATH,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Evaluate a trained SAC model on random windows.
    
    Returns:
        - episode_stats: list of dicts with per-episode metrics
        - trajectory_df: DataFrame with per-step info from all episodes
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}.zip")
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from {vecnormalize_path}")
    else:
        vecnormalize_path = None
        print("VecNormalize stats not found; evaluating on raw observations.")

    # Setup env kwargs
    kwargs = {
        "forecast_h": forecast_h,
        "price_scale": price_scale,
        "wind_scale": wind_scale,
        "enforce_cyclic_boundary": True,
    }
    if env_kwargs:
        kwargs.update(env_kwargs)

    episode_stats = []
    trajectory_list = []

    np.random.seed(seed)

    def _post_bypass_violation(step_info: dict[str, Any]) -> float:
        beta1 = float(step_info.get("beta1", step_info.get("beta1_k", 0.0)))
        beta2 = float(step_info.get("beta2", step_info.get("beta2_k", 0.0)))
        return max(1.0 - (beta1 + beta2), 0.0)

    def _post_complementarity_violation(step_info: dict[str, Any]) -> float:
        q_ch = float(step_info.get("Q_s_ch_k", step_info.get("Q_ch", 0.0)))
        q_dch = float(step_info.get("Q_s_dch_k", step_info.get("Q_dch", 0.0)))
        return max(min(q_ch, q_dch), 0.0)

    def _post_grid_violation(step_info: dict[str, Any]) -> float:
        p_grid = float(step_info.get("P_grid_k", step_info.get("P_grid", 0.0)))
        p_max = float(step_info.get("P_grid_max", 5000.0))
        return max(-p_grid, 0.0) + max(p_grid - p_max, 0.0)

    for ep in range(num_episodes):
        # Sample random window
        n = len(g_grid_full)
        max_start = n - window_len
        start = np.random.randint(0, max_start + 1)
        end = start + window_len

        raw_env = SteamEnv(
            g_grid_full[start:end],
            P_WT_full[start:end],
            **kwargs,
        )

        env = DummyVecEnv([lambda raw_env=raw_env: raw_env])
        if vecnormalize_path:
            env = VecNormalize.load(vecnormalize_path, env)
            env.training = False
            env.norm_reward = False

        obs = env.reset()
        done = False
        step = 0
        cumulative_reward = 0.0

        # Per-step tracking
        step_info_list = []

        while not done:
            # Deterministic action
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, infos = env.step(action)
            done = bool(terminated[0])
            step_info = infos[0]
            cumulative_reward += float(reward[0])
            step += 1

            # Log step info
            step_info_list.append({
                "episode": ep,
                "step": step,
                "action_repaired": bool(step_info.get("action_repaired", False)),
                "action_beta1": step_info.get("beta1_k"),
                "action_beta2": step_info.get("beta2_k"),
                "action_beta1_raw": step_info.get("beta1_raw"),
                "action_beta2_raw": step_info.get("beta2_raw"),
                "action_R": step_info.get("R_k"),
                "action_m_I": step_info.get("m_I_k"),
                "action_T_I": step_info.get("T_I_k"),
                "f4_projection_applied": bool(step_info.get("f4_projection_applied_k", False)),
                "f4_projection_feasible": step_info.get("f4_projection_feasible_k"),
                "f4_infeasible_reasons": step_info.get("f4_infeasible_reasons_k"),
                "f4_initial_infeasible_reasons": step_info.get("f4_initial_infeasible_reasons_k"),
                "f4_mode_repair_applied": bool(step_info.get("f4_mode_repair_applied_k", False)),
                "f4_mode_candidate": step_info.get("f4_mode_candidate_k"),
                "reward": float(reward[0]),
                "cost_grid": step_info.get("cost_grid_k"),
                "P_grid": step_info.get("P_grid_k"),
                "P_spill": step_info.get("P_spill_k"),
                "P_HP": step_info.get("P_HP_k"),
                "P_WT": step_info.get("P_WT_k"),
                "T_s": step_info.get("T_s_k"),
                "Q_ch": step_info.get("Q_s_ch_k"),
                "Q_dch": step_info.get("Q_s_dch_k"),
                "mode": step_info.get("mode_k"),
                "constraint_violation_bypass_post": _post_bypass_violation(step_info),
                "constraint_violation_complementarity_post": _post_complementarity_violation(step_info),
                "constraint_violation_grid_post": _post_grid_violation(step_info),
                "constraint_violation_bypass": step_info.get("bypass_sum_violation_k"),
                "constraint_violation_complementarity": step_info.get("complementarity_violation_k"),
                "constraint_violation_state": step_info.get("state_violation_k"),
                "constraint_violation_grid": step_info.get("grid_bound_violation_k"),
                "constraint_violation_f4": abs(step_info.get("f4_residual_k", 0.0)),
                "constraint_violation_f5": abs(step_info.get("f5_residual_k", 0.0)),
                "constraint_violation_coupling": abs(step_info.get("f4_residual_k", 0.0)) + abs(step_info.get("f5_residual_k", 0.0)),
            })

        trajectory_df_ep = pd.DataFrame(step_info_list)
        trajectory_list.append(trajectory_df_ep)

        env.close()

        # Aggregate episode stats
        total_violations = (
            trajectory_df_ep["constraint_violation_bypass_post"].sum()
            + trajectory_df_ep["constraint_violation_complementarity_post"].sum()
            + trajectory_df_ep["constraint_violation_state"].sum()
            + trajectory_df_ep["constraint_violation_grid_post"].sum()
            + trajectory_df_ep["constraint_violation_coupling"].sum()
        )

        ep_stats = {
            "episode": ep,
            "start_idx": start,
            "cumulative_reward": cumulative_reward,
            "avg_reward": cumulative_reward / step,
            "num_steps": step,
            "total_cost": trajectory_df_ep["cost_grid"].sum(),
            "avg_cost": trajectory_df_ep["cost_grid"].mean(),
            "total_constraint_violation": total_violations,
            "avg_P_grid": trajectory_df_ep["P_grid"].mean(),
            "std_P_grid": trajectory_df_ep["P_grid"].std(),
            "max_P_grid": trajectory_df_ep["P_grid"].max(),
            "min_P_grid": trajectory_df_ep["P_grid"].min(),
            "avg_T_s": trajectory_df_ep["T_s"].mean(),
            "min_T_s": trajectory_df_ep["T_s"].min(),
            "max_T_s": trajectory_df_ep["T_s"].max(),
            "avg_Q_ch": trajectory_df_ep["Q_ch"].mean(),
            "avg_Q_dch": trajectory_df_ep["Q_dch"].mean(),
            "charging_steps": (trajectory_df_ep["Q_ch"] > 1e-3).sum(),
            "discharging_steps": (trajectory_df_ep["Q_dch"] > 1e-3).sum(),
            "repair_steps": trajectory_df_ep["action_repaired"].sum(),
            "f4_projection_steps": trajectory_df_ep["f4_projection_applied"].sum(),
            "f4_infeasible_steps": (trajectory_df_ep["f4_projection_feasible"] == False).sum(),
            "f4_mode_repair_steps": trajectory_df_ep["f4_mode_repair_applied"].sum(),
        }
        episode_stats.append(ep_stats)
        
        print(
            f"Episode {ep}: reward={cumulative_reward:.2f}, "
            f"cost={ep_stats['total_cost']:.2f}, violations={total_violations:.2f}, steps={step}"
        )

    # Combine all trajectories
    all_trajectories = pd.concat(trajectory_list, ignore_index=True) if trajectory_list else pd.DataFrame()

    return episode_stats, all_trajectories


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC model on Cookie Factory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to saved SAC model")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV or Excel dataset")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--window-len", type=int, default=24, help="Random window length")
    parser.add_argument("--forecast-h", type=int, default=24, help="Forecast horizon")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic (random) actions instead of deterministic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="results/eval", help="Where to save results")
    parser.add_argument("--price-col", type=str, default="Pe", help="Column name for grid price")
    parser.add_argument("--wind-col", type=str, default="WKA", help="Column name for wind power")
    parser.add_argument("--first-24", action="store_true", help="Evaluate exactly the first 24 rows of the dataset")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.first_24:
        evaluate_first_24_data(
            model_path=args.model_path,
            data_path=args.data,
            price_col=args.price_col,
            wind_col=args.wind_col,
            forecast_h=args.forecast_h,
            deterministic=not args.stochastic,
            seed=args.seed,
            vecnormalize_path=f"{args.model_path}_vecnormalize.pkl",
        )
        return

    g_grid_full, P_WT_full, price_scale, wind_scale = load_cookie_data(
        data_path=args.data,
        price_col=args.price_col,
        wind_col=args.wind_col,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating model: {args.model_path}")
    print(f"Episodes: {args.num_episodes}, Window length: {args.window_len}")
    print(f"Deterministic: {not args.stochastic}")
    print(f"{'='*60}\n")

    episode_stats, trajectories = evaluate_model(
        model_path=args.model_path,
        g_grid_full=g_grid_full,
        P_WT_full=P_WT_full,
        price_scale=price_scale,
        wind_scale=wind_scale,
        num_episodes=args.num_episodes,
        window_len=args.window_len,
        forecast_h=args.forecast_h,
        deterministic=not args.stochastic,
        seed=args.seed,
        vecnormalize_path=f"{args.model_path}_vecnormalize.pkl",
    )

    # Save episode summary
    episode_df = pd.DataFrame(episode_stats)
    episode_csv = os.path.join(args.output_dir, "eval_episodes.csv")
    episode_df.to_csv(episode_csv, index=False)
    print(f"\nSaved episode summary to {episode_csv}")
    print(episode_df.to_string(index=False))

    # Save detailed trajectories
    if not trajectories.empty:
        traj_csv = os.path.join(args.output_dir, "eval_trajectories.csv")
        trajectories.to_csv(traj_csv, index=False)
        print(f"\nSaved trajectory details to {traj_csv}")

        violation_columns = [
            "constraint_violation_bypass_post",
            "constraint_violation_complementarity_post",
            "constraint_violation_state",
            "constraint_violation_grid_post",
            "constraint_violation_coupling",
        ]
        violation_summary = trajectories[violation_columns].sum().sort_values(ascending=False).reset_index()
        violation_summary.columns = ["constraint", "total_violation"]
        violation_summary["mean_violation_per_step"] = violation_summary["total_violation"] / len(trajectories)

        violation_csv = os.path.join(args.output_dir, "eval_violation_breakdown.csv")
        violation_summary.to_csv(violation_csv, index=False)
        print(f"Saved violation breakdown to {violation_csv}")
        print("\nViolation breakdown:")
        print(violation_summary.to_string(index=False))

    # Print summary stats
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Mean cumulative reward:    {episode_df['cumulative_reward'].mean():.4f}")
    print(f"Std cumulative reward:     {episode_df['cumulative_reward'].std():.4f}")
    print(f"Mean total cost:           {episode_df['total_cost'].mean():.4f}")
    print(f"Mean avg cost/step:        {episode_df['avg_cost'].mean():.4f}")
    print(f"Mean constraint violation: {episode_df['total_constraint_violation'].mean():.4f}")
    print(f"Max constraint violation:  {episode_df['total_constraint_violation'].max():.4f}")
    print(f"Mean avg P_grid (W):       {episode_df['avg_P_grid'].mean():.2f}")
    print(f"Mean storage temp (C):     {episode_df['avg_T_s'].mean():.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
