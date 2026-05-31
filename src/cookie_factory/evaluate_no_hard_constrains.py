"""Simple evaluator for the soft-constrained Cookie Factory SAC model.

Run:
python -m src.cookie_factory.evaluate_no_hard_constrains --data cookie_model/WKA_Pe_merged.csv --model-path src/cookie_factory/cookie_sac_model_no_hard_constrains --window-len 24 --forecast-h 24
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.cookie_factory.cookie_factory_env_no_hard_constrains import SteamEnv
from src.cookie_factory.train_no_hard_constrains import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    load_cookie_data,
)


VIOLATION_KEYS = (
    "action_bounds",
    "bypass_sum",
    "complementarity",
    "q_ch_lower",
    "q_dch_lower",
    "q_ch_upper",
    "q_dch_upper",
    "storage_lower",
    "storage_upper",
    "grid_lower",
    "grid_upper",
    "f4_residual",
    "f5_residual",
    "terminal_storage",
)


def constraint_diagnostics(env: SteamEnv, action: np.ndarray) -> dict[str, float]:
    """Compute the same soft constraint violations used by the environment."""
    action = np.asarray(action, dtype=np.float32)
    clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
    action_span = env.action_space.high - env.action_space.low
    beta1, beta2, R, m_I, T_I = (float(value) for value in clipped_action)

    k = env.k
    T_0 = env.T_s_k
    T_2, T_3 = env._steam_generator_temperatures(m_I)
    T_4 = T_3 - env.epsilon_dch * (T_3 - T_0)
    P_HP, T_II = env._hthp_surrogate(T_I, m_I, R)
    T_1 = T_II - env.epsilon_ch * (T_II - T_0)

    Q_ch = (T_II - T_1) * m_I * 3 * env.c_p_f * (1.0 - beta1)
    Q_dch = (T_4 - T_3) * m_I * 3 * env.c_p_f * (1.0 - beta2)
    P_grid = P_HP - float(env.P_WT[k])
    T_next = T_0 + (Q_ch - Q_dch) * env.Delta_t / max(env.m_s * env.c_p_s, 1e-6)

    diagnostics = {
        "action_bounds": float(np.sum(np.abs(action - clipped_action) / action_span)),
        "bypass_sum": max(1.0 - beta1 - beta2, 0.0),
        "complementarity": max(
            max(Q_ch, 0.0) * max(Q_dch, 0.0) - env.eps_comp,
            0.0,
        ),
        "q_ch_lower": max(-Q_ch, 0.0),
        "q_dch_lower": max(-Q_dch, 0.0),
        "q_ch_upper": max(Q_ch - env.Q_flow_max, 0.0),
        "q_dch_upper": max(Q_dch - env.Q_flow_max, 0.0),
        "storage_lower": max(env.T_s_min - T_next, 0.0),
        "storage_upper": max(T_next - env.T_s_max, 0.0),
        "grid_lower": max(-P_grid, 0.0),
        "grid_upper": max(P_grid - env.P_HP_max, 0.0),
        "f4_residual": abs(T_2 - (T_1 * (1.0 - beta1) + T_II * beta1)),
        "f5_residual": abs(T_I - (T_3 * beta2 + T_4 * (1.0 - beta2))),
        "terminal_storage": 0.0,
        "P_grid": float(P_grid),
        "T_s_next": float(T_next),
        "Q_ch": float(Q_ch),
        "Q_dch": float(Q_dch),
        "beta1": beta1,
        "beta2": beta2,
        "R": R,
        "m_I": m_I,
        "T_I": T_I,
    }
    if k + 1 >= env.K and env.enforce_cyclic_boundary:
        diagnostics["terminal_storage"] = abs(T_next - env.T_s_0)
    return diagnostics


def print_step_violations(step: int, diagnostics: dict[str, float], tol: float) -> None:
    violated = {
        key: diagnostics[key]
        for key in VIOLATION_KEYS
        if diagnostics[key] > tol
    }
    if not violated:
        print(f"step {step:02d}: no constraint violation")
        return

    parts = ", ".join(f"{key}={value:.6g}" for key, value in violated.items())
    print(f"step {step:02d}: {parts}")


def evaluate(args: argparse.Namespace) -> None:
    if not os.path.exists(f"{args.model_path}.zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    g_grid, P_WT, price_scale, wind_scale = load_cookie_data(
        data_path=args.data,
        price_col=args.price_col,
        wind_col=args.wind_col,
    )
    if args.start_idx + args.window_len > len(g_grid):
        raise ValueError("start_idx + window_len is larger than the dataset")

    raw_env = SteamEnv(
        g_grid[args.start_idx : args.start_idx + args.window_len],
        P_WT[args.start_idx : args.start_idx + args.window_len],
        forecast_h=args.forecast_h,
        price_scale=price_scale,
        wind_scale=wind_scale,
        enforce_cyclic_boundary=not args.no_cyclic_boundary,
    )
    env = DummyVecEnv([lambda: raw_env])

    vecnormalize_path = args.vecnormalize_path or f"{args.model_path}_vecnormalize.pkl"
    if os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from {vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("VecNormalize stats not found; evaluating on raw observations.")

    model = SAC.load(args.model_path)
    obs = env.reset()
    done = False
    step = 0
    cumulative_reward = 0.0
    totals = {key: 0.0 for key in VIOLATION_KEYS}

    print(f"\nEvaluating {args.model_path}.zip on rows {args.start_idx}:{args.start_idx + args.window_len}")
    print("Constraint violations:")

    while not done:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        diagnostics = constraint_diagnostics(raw_env, action[0])
        obs, reward, dones, _ = env.step(action)

        step += 1
        done = bool(dones[0])
        cumulative_reward += float(reward[0])
        for key in VIOLATION_KEYS:
            totals[key] += diagnostics[key]
        print_step_violations(step, diagnostics, args.tolerance)

    env.close()

    print("\nViolation summary:")
    any_violation = False
    for key, value in totals.items():
        if value > args.tolerance:
            any_violation = True
            print(f"{key:20s} total={value:.6g} avg={value / step:.6g}")
    if not any_violation:
        print("No constraint violations above tolerance.")
    print(f"\nCumulative reward: {cumulative_reward:.6g}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate soft-constrained Cookie Factory SAC model")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to saved SAC model without .zip")
    parser.add_argument("--vecnormalize-path", default=None, help="Optional VecNormalize .pkl path")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV or Excel dataset")
    parser.add_argument("--window-len", type=int, default=24, help="Evaluation window length")
    parser.add_argument("--start-idx", type=int, default=0, help="First row of the evaluation window")
    parser.add_argument("--forecast-h", type=int, default=24, help="Forecast horizon in env")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic SAC actions")
    parser.add_argument("--no-cyclic-boundary", action="store_true", help="Disable terminal storage penalty/violation")
    parser.add_argument("--price-col", default="Pe", help="Column name for grid price")
    parser.add_argument("--wind-col", default="WKA", help="Column name for wind power")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Only print violations above this value")
    return parser.parse_args()


def main() -> None:
    evaluate(_parse_args())


if __name__ == "__main__":
    main()
