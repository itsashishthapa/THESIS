"""Compare a Cookie Factory RL policy with the Mathematical optimisation model.

Both methods are evaluated on the same fixed or randomly sampled 24-step
price/wind windows.  The reported objective is grid-electricity cost in EUR,
which is the common objective used by ``SteamEnv`` and ``cookie_optimization``.

Example:
    python -m src.cookie_factory.compare_models \
        --algorithm SAC --model-path src/cookie_factory/cookie_sac_derived \
        --data cookie_model/WKA_Pe_merged.csv --comparisons 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cookie_model.cookie_optimization import get_cookie_model
from src.cookie_factory.cookie_factory_env import SteamEnv
from src.cookie_factory.train import (
    DEFAULT_ALGORITHM,
    get_model_class,
    get_model_path,
    load_cookie_data,
    load_environment_config,
    normalize_algorithm,
)

DEFAULT_MODEL_PATH = "src/cookie_factory/cookie_sac_derived"
DEFAULT_DATA_PATH = "cookie_model/WKA_Pe_merged.csv"
DEFAULT_OUTPUT_DIR = "results/cookie_factory_comparison"
VIOLATION_TOLERANCE = 1e-6
TERMINAL_STORAGE_TEMPERATURE = 250.0
GUARANTEED_RL_CONSTRAINTS = frozenset({"bypass", "complementarity", "f5"})


def _constraint_violations(info: dict[str, Any]) -> dict[str, float]:
    """Calculate comparable per-step constraint-violation magnitudes."""
    beta1 = float(info["beta1"])
    beta2 = float(info["beta2"])
    q_ch = float(info["Q_ch"])
    q_dch = float(info["Q_dch"])
    p_grid = float(info["P_grid"])
    t_s = float(info["T_s"])
    t_i = float(info["T_I"])

    f4 = abs(float(info["f4_residual"]))
    f5 = abs(float(info["f5_residual"]))
    violations = {
        "bypass": max(1.0 - beta1 - beta2, 0.0),
        "complementarity": max(min(q_ch, q_dch), 0.0),
        "flow": (
            max(-q_ch, 0.0)
            + max(q_ch - 5000.0, 0.0)
            + max(-q_dch, 0.0)
            + max(q_dch - 5000.0, 0.0)
        ),
        "inlet_temperature": max(177.0 - t_i, 0.0) + max(t_i - 250.0, 0.0),
        "state": max(183.0 - t_s, 0.0) + max(t_s - 324.0, 0.0),
        "grid": max(-p_grid, 0.0) + max(p_grid - 5000.0, 0.0),
        "f4": f4,
        "f5": f5,
    }
    violations["total"] = float(sum(violations.values()))
    return violations


def _rl_step_info(info: dict[str, Any]) -> dict[str, float]:
    """Convert a SteamEnv info dictionary to a method-independent trajectory row."""
    return {
        "P_grid": float(info["P_grid_k"]),
        "T_s": float(info["T_s_k"]),
        "T_I": float(info["T_I_k"]),
        "Q_ch": float(info["Q_s_ch_k"]),
        "Q_dch": float(info["Q_s_dch_k"]),
        "beta1": float(info["beta1_k"]),
        "beta2": float(info["beta2_k"]),
        "R": float(info["R_k"]),
        "m_I": float(info["m_I_k"]),
        "cost": float(info["cost_grid_k"]),
        "f4_residual": float(info["f4_residual_k"]),
        "f5_residual": 0.0,
    }


def _add_terminal_cyclic_violation(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Record the cyclic TES-temperature residual on the final rollout step."""
    if trajectory.empty:
        return trajectory

    trajectory = trajectory.copy()
    terminal_deviation = abs(
        float(trajectory.iloc[-1]["T_s"]) - TERMINAL_STORAGE_TEMPERATURE
    )
    trajectory["violation_terminal_cyclic"] = 0.0
    trajectory.loc[trajectory.index[-1], "violation_terminal_cyclic"] = (
        terminal_deviation if terminal_deviation > VIOLATION_TOLERANCE else 0.0
    )
    trajectory["violation_total"] += trajectory["violation_terminal_cyclic"]
    return trajectory


def _violation_totals(trajectory: pd.DataFrame) -> dict[str, float]:
    """Return a total for every individual violation recorded in a trajectory."""
    return {
        column.removeprefix("violation_"): float(trajectory[column].sum())
        for column in trajectory.columns
        if column.startswith("violation_")
    }


def run_rl_model(
    model: SAC | TD3,
    vecnormalize_path: str | None,
    prices: np.ndarray,
    wind: np.ndarray,
    price_scale: float,
    wind_scale: float,
    forecast_h: int,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one deterministic RL rollout and return cost and violations."""
    kwargs = dict(forecast_h=forecast_h, price_scale=price_scale,
                  wind_scale=wind_scale, enforce_cyclic_boundary=True)
    kwargs.update(env_kwargs or {})
    raw_env = SteamEnv(prices, wind, **kwargs)
    env: DummyVecEnv | VecNormalize = DummyVecEnv([lambda: raw_env])
    if vecnormalize_path is not None:
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False

    started = time.perf_counter()
    observation = env.reset()
    done = False
    rows: list[dict[str, float]] = []

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, infos = env.step(action)
        info = infos[0]
        row = _rl_step_info(info)
        violations = _constraint_violations(
            {
                "beta1": row["beta1"],
                "beta2": row["beta2"],
                "Q_ch": row["Q_ch"],
                "Q_dch": row["Q_dch"],
                "P_grid": row["P_grid"],
                "T_s": row["T_s"],
                "T_I": row["T_I"],
                "f4_residual": row["f4_residual"],
                "f5_residual": row["f5_residual"],
            }
        )
        row.update({f"violation_{name}": value for name, value in violations.items()})
        row["reward"] = float(reward[0])
        rows.append(row)
        done = bool(terminated[0])

    elapsed = time.perf_counter() - started
    env.close()
    trajectory = _add_terminal_cyclic_violation(pd.DataFrame(rows))
    violation_totals = _violation_totals(trajectory)
    return (
        {
            "cost": float(trajectory["cost"].sum()),
            "time_s": elapsed,
            "constraint_violation": violation_totals["total"],
            "violation_totals": violation_totals,
            "reward": float(trajectory["reward"].sum()),
        },
        trajectory,
    )


def run_pyomo_model(
    prices: np.ndarray,
    wind: np.ndarray,
    solver_name: str,
    max_iter: int,
) -> tuple[dict[str, Any] | None, pd.DataFrame | None, str | None]:
    """Solve one matching Pyomo window and calculate its feasibility metrics."""
    result = get_cookie_model(
        prices,
        wind,
        solver_name=solver_name,
        max_iter=max_iter,
        verbose=False,
    )
    if result["status"] != "success":
        return None, None, result.get("error", "Pyomo solve failed")

    trajectory = result["trajectory"].copy()
    trajectory.rename(
        columns={
            "P_grid": "P_grid",
            "T_s": "T_s",
            "Q_ch": "Q_ch",
            "Q_dch": "Q_dch",
            "T_I": "T_I",
            "m_I": "m_I",
            "R": "R",
            "T_II": "T_II",
            "T_1": "T_1",
            "T_2": "T_2",
            "T_3": "T_3",
            "T_4": "T_4",
        },
        inplace=True,
    )
    trajectory["cost"] = prices / 1000.0 * trajectory["P_grid"]
    trajectory["f4_residual"] = np.abs(
        trajectory["T_2"]
        - (trajectory["T_1"] * (1.0 - trajectory["beta1"])
           + trajectory["T_II"] * trajectory["beta1"])
    )
    trajectory["f5_residual"] = np.abs(
        trajectory["T_I"]
        - (trajectory["T_3"] * trajectory["beta2"]
           + trajectory["T_4"] * (1.0 - trajectory["beta2"]))
    )

    violations = trajectory.apply(
        lambda row: _constraint_violations(row.to_dict()), axis=1, result_type="expand"
    )
    violations.columns = [f"violation_{name}" for name in violations.columns]
    trajectory = _add_terminal_cyclic_violation(pd.concat([trajectory, violations], axis=1))
    violation_totals = _violation_totals(trajectory)
    return (
        {
            "cost": float(result["cost"]),
            "time_s": float(result["time"]),
            "constraint_violation": violation_totals["total"],
            "violation_totals": violation_totals,
            "reward": float("nan"),
        },
        trajectory,
        None,
    )


def compare_models(
    model_path: str,
    algorithm: str,
    data_path: str,
    comparisons: int,
    window_len: int,
    forecast_h: int,
    seed: int,
    solver_name: str,
    max_iter: int,
    start_hour: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare an RL policy and Pyomo over identical input windows."""
    if comparisons <= 0:
        raise ValueError("comparisons must be positive")
    if forecast_h > window_len:
        raise ValueError("forecast_h cannot be larger than window_len")
    if not Path(f"{model_path}.zip").exists():
        raise FileNotFoundError(f"RL model not found: {model_path}.zip")

    prices, wind, price_scale, wind_scale = load_cookie_data(data_path)
    if len(prices) < window_len:
        raise ValueError("The data contains fewer rows than window_len")
    max_start = len(prices) - window_len
    if start_hour is not None and not 0 <= start_hour <= max_start:
        raise ValueError(f"start_hour must be between 0 and {max_start}")

    vecnormalize_path = f"{model_path}_vecnormalize.pkl"
    if not Path(vecnormalize_path).exists():
        vecnormalize_path = None
        print("VecNormalize statistics not found; using raw observations.")

    algorithm = normalize_algorithm(algorithm)
    model = get_model_class(algorithm).load(model_path)
    environment_config = load_environment_config(model_path)
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, Any]] = []
    trajectory_frames: list[pd.DataFrame] = []

    for comparison in range(comparisons):
        start = start_hour if start_hour is not None else int(rng.integers(max_start + 1))
        end = start + window_len
        prices_window = prices[start:end]
        wind_window = wind[start:end]
        print(f"Comparison {comparison + 1}/{comparisons}: rows {start}:{end}")

        rl_metrics, rl_trajectory = run_rl_model(
            model,
            vecnormalize_path,
            prices_window,
            wind_window,
            price_scale,
            wind_scale,
            forecast_h,
            env_kwargs=environment_config,
        )
        pyomo_metrics, pyomo_trajectory, error = run_pyomo_model(
            prices_window,
            wind_window,
            solver_name,
            max_iter,
        )
        if pyomo_metrics is None or pyomo_trajectory is None:
            print(f"  Pyomo failed: {error}")
            summary_rows.append(
                {
                    "comparison": comparison,
                    "start_idx": start,
                    "status": "pyomo_failed",
                    "error": error,
                }
            )
            continue

        cost_gap = rl_metrics["cost"] - pyomo_metrics["cost"]
        comparison_row = {
            "comparison": comparison,
            "start_idx": start,
            "status": "success",
            "rl_cost": rl_metrics["cost"],
            "pyomo_cost": pyomo_metrics["cost"],
            "cost_gap": cost_gap,
            "relative_cost_gap_pct": 100.0 * cost_gap / max(abs(pyomo_metrics["cost"]), 1e-12),
            "rl_time_s": rl_metrics["time_s"],
            "pyomo_time_s": pyomo_metrics["time_s"],
            "rl_constraint_violation": rl_metrics["constraint_violation"],
            "pyomo_constraint_violation": pyomo_metrics["constraint_violation"],
            "rl_reward": rl_metrics["reward"],
        }
        comparison_row.update(
            {
                f"rl_violation_{name}": value
                for name, value in rl_metrics["violation_totals"].items()
            }
        )
        comparison_row.update(
            {
                f"pyomo_violation_{name}": value
                for name, value in pyomo_metrics["violation_totals"].items()
            }
        )
        summary_rows.append(comparison_row)
        for method, evaluated_model, trajectory in (
            ("rl", algorithm, rl_trajectory),
            ("pyomo", "Pyomo", pyomo_trajectory),
        ):
            trajectory = trajectory.copy()
            trajectory.insert(0, "evaluated_model", evaluated_model)
            trajectory.insert(0, "method", method)
            trajectory.insert(0, "start_idx", start)
            trajectory.insert(0, "comparison", comparison)
            trajectory_frames.append(trajectory)
        print(
            f"  RL cost={rl_metrics['cost']:.2f} EUR ({rl_metrics['time_s']:.3f}s), "
            f"Pyomo cost={pyomo_metrics['cost']:.2f} EUR ({pyomo_metrics['time_s']:.3f}s)"
        )

    summary = pd.DataFrame(summary_rows)
    trajectories = (
        pd.concat(trajectory_frames, ignore_index=True) if trajectory_frames else pd.DataFrame()
    )
    return summary, trajectories


def summary_text(summary: pd.DataFrame, algorithm: str) -> str:
    """Build a human-readable matched-window comparison report."""
    successful = summary.loc[summary["status"] == "success"].copy()
    if successful.empty:
        return "No successful RL/Pyomo comparisons were completed.\n"

    speedup = successful["pyomo_time_s"].mean() / max(successful["rl_time_s"].mean(), 1e-12)
    lines = [
        "=" * 80,
        f"COMPARISON: Cookie Factory {algorithm} Policy vs Pyomo Optimisation",
        "=" * 80,
        f"Successful matched windows: {len(successful)}",
        "",
        "COST COMPARISON (EUR)",
        "-" * 80,
        f"{algorithm} mean cost:   {successful['rl_cost'].mean():.4f} +/- {successful['rl_cost'].std():.4f}",
        f"Pyomo mean cost: {successful['pyomo_cost'].mean():.4f} +/- {successful['pyomo_cost'].std():.4f}",
        f"Mean {algorithm} - Pyomo gap: {successful['cost_gap'].mean():.4f} EUR",
        f"Mean absolute relative gap: {successful['relative_cost_gap_pct'].abs().mean():.2f}%",
        "",
        "COMPUTATION TIME (SECONDS)",
        "-" * 80,
        f"{algorithm} mean time:   {successful['rl_time_s'].mean():.4f}",
        f"Pyomo mean time: {successful['pyomo_time_s'].mean():.4f}",
        f"Speedup (Pyomo / {algorithm}): {speedup:.2f}x",
        "",
        "CONSTRAINT VIOLATIONS",
        "-" * 80,
        f"{algorithm} mean total violation:   {successful['rl_constraint_violation'].mean():.4f}",
        f"Pyomo mean total violation: {successful['pyomo_constraint_violation'].mean():.4e}",
    ]
    violation_names = {
        column.removeprefix("rl_violation_")
        for column in successful.columns
        if column.startswith("rl_violation_")
        and column != "rl_violation_total"
        and f"pyomo_violation_{column.removeprefix('rl_violation_')}" in successful
    }
    violation_names = sorted(violation_names - GUARANTEED_RL_CONSTRAINTS)
    if violation_names:
        lines.extend(["", "VIOLATION BREAKDOWN (MEAN TOTAL PER WINDOW)", "-" * 80])
        for name in violation_names:
            label = name.replace("_", " ")
            lines.append(
                f"{label}: {algorithm}={successful[f'rl_violation_{name}'].mean():.4e}, "
                f"Pyomo={successful[f'pyomo_violation_{name}'].mean():.4e}"
            )
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def print_summary(
    summary: pd.DataFrame, algorithm: str, output_path: Path | None = None
) -> None:
    """Print and optionally save the human-readable comparison report."""
    report = summary_text(summary, algorithm)
    print(f"\n{report}")
    if output_path is not None:
        output_path.write_text(report, encoding="utf-8")


def plot_summary(summary: pd.DataFrame, algorithm: str, output_path: Path) -> None:
    """Save matched-window cost, runtime, and violation comparisons."""
    import matplotlib.pyplot as plt

    successful = summary.loc[summary["status"] == "success"].copy()
    if successful.empty:
        return

    x = np.arange(len(successful))
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(x, successful["rl_cost"], marker="o", label=algorithm)
    axes[0].plot(x, successful["pyomo_cost"], marker="o", label="Pyomo")
    axes[0].set_ylabel("Cost [EUR]")
    axes[0].set_title(f"Cookie Factory: {algorithm} vs Pyomo on matched windows")
    axes[0].legend()

    axes[1].plot(x, successful["rl_time_s"], marker="o", label=algorithm)
    axes[1].plot(x, successful["pyomo_time_s"], marker="o", label="Pyomo")
    axes[1].set_ylabel("Runtime [s]")
    axes[1].legend()

    axes[2].plot(x, successful["rl_constraint_violation"], marker="o", label=algorithm)
    axes[2].plot(x, successful["pyomo_constraint_violation"], marker="o", label="Pyomo")
    axes[2].set_ylabel("Total violation")
    axes[2].set_xlabel("Comparison window")
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def violation_breakdown(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Aggregate violations not guaranteed by the RL environment construction."""
    violation_columns = [
        column
        for column in trajectories.columns
        if column.startswith("violation_")
        and column.removeprefix("violation_") not in GUARANTEED_RL_CONSTRAINTS
    ]
    if trajectories.empty or not violation_columns:
        return pd.DataFrame(
            columns=[
                "comparison", "start_idx", "method", "evaluated_model", "constraint",
                "total_violation", "mean_violation_per_step", "violating_steps",
            ]
        )

    identifiers = ["comparison", "start_idx", "method", "evaluated_model"]
    long_form = trajectories.melt(
        id_vars=identifiers,
        value_vars=violation_columns,
        var_name="constraint",
        value_name="violation",
    )
    long_form["constraint"] = long_form["constraint"].str.removeprefix("violation_")
    return (
        long_form.groupby(identifiers + ["constraint"], as_index=False)
        .agg(
            total_violation=("violation", "sum"),
            mean_violation_per_step=("violation", "mean"),
            violating_steps=(
                "violation",
                lambda values: int((values > VIOLATION_TOLERANCE).sum()),
            ),
        )
        .sort_values(identifiers + ["constraint"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Cookie Factory RL and Pyomo")
    parser.add_argument(
        "--algorithm",
        type=normalize_algorithm,
        default=DEFAULT_ALGORITHM,
        choices=("SAC", "TD3"),
        help="RL algorithm used by the saved model",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--comparisons", type=int, default=10)
    parser.add_argument("--window-len", type=int, default=24)
    parser.add_argument("--forecast-h", type=int, default=24)
    parser.add_argument("--start-hour", type=int, default=None)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--solver", default="ipopt")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for model-specific comparison files "
            "(default: results/cookie_factory_comparison)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path or get_model_path(args.algorithm)
    summary, trajectories = compare_models(
        model_path=args.model_path,
        algorithm=args.algorithm,
        data_path=args.data,
        comparisons=args.comparisons,
        window_len=args.window_len,
        forecast_h=args.forecast_h,
        seed=args.seed,
        solver_name=args.solver,
        max_iter=args.max_iter,
        start_hour=args.start_hour,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_stem = f"model_comparison_{args.algorithm.lower()}"
    summary.to_csv(output_dir / f"{filename_stem}_summary.csv", index=False)
    if not trajectories.empty:
        trajectories.to_csv(output_dir / f"{filename_stem}_trajectories.csv", index=False)
    violations = violation_breakdown(trajectories)
    violations.to_csv(output_dir / f"{filename_stem}_violations.csv", index=False)
    metadata = {
        "evaluated_model": args.algorithm,
        "model_path": args.model_path,
        "pyomo_solver": args.solver,
        "comparisons_requested": args.comparisons,
        "window_len": args.window_len,
        "forecast_h": args.forecast_h,
        "violation_tolerance": VIOLATION_TOLERANCE,
    }
    (output_dir / f"{filename_stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    plot_summary(summary, args.algorithm, output_dir / f"{filename_stem}.png")
    print_summary(summary, args.algorithm, output_dir / f"{filename_stem}.txt")
    print(f"\nSaved comparison files to {output_dir}")


if __name__ == "__main__":
    main()
