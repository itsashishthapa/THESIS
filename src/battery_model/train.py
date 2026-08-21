"""
Training script for Battery RL model using SAC or TD3.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# from Battery_model.Optimization_Pyomo.pyomo_battery_model import optimize_battery_pyomo
from src.battery_model.battery_env import RandomWindowEnv
from src.battery_model.evaluate import (
    DEFAULT_ALGORITHM,
    compute_total_cost,
    evaluate_model,
    get_model_class,
    get_model_path,
    normalize_algorithm,
)
from src.battery_model.price_model import (
    DEFAULT_PRICE_MODEL_HOURS,
    load_price_model_data,
)

# from src.battery_model.terminal_replay_buffer import TerminalStratifiedReplayBuffer


def load_price_data():
    """Load electricity price data from file"""
    price_df = pd.read_csv(
        "./Battery_model/Optimization_Pyomo/input_data/electricity_price.txt",
        sep=r"\s+",
        header=None,
        names=["t", "price"],
    )
    prices_full = price_df["price"].values.astype(float)

    # Global scaler (95th percentile of absolute prices across full dataset)
    global_price_scale = np.percentile(np.abs(prices_full), 95) + 1e-6

    print(f"Full dataset length: {len(prices_full)} hours")
    print(f"Global price scale (95th pct): {global_price_scale:.4f}")

    return prices_full, global_price_scale


def make_train_env(prices_full, price_scale, log_dir):
    """Create a training environment with random price windows"""
    env = RandomWindowEnv(prices_full, window_len=24, price_scale=price_scale)
    env = Monitor(env)
    return env


def split_train_validation_prices(
    prices_full,
    validation_fraction=0.2,
    window_len=24,
):
    """Split a price series into chronological training and validation blocks."""
    prices_full = np.asarray(prices_full, dtype=float)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    split_idx = int(len(prices_full) * (1.0 - validation_fraction))
    prices_train = prices_full[:split_idx]
    prices_validation = prices_full[split_idx:]

    if len(prices_train) < window_len or len(prices_validation) < window_len:
        raise ValueError(
            "Training and validation blocks must each contain at least "
            f"{window_len} prices."
        )

    return prices_train, prices_validation


class BestValidationCostCallback(BaseCallback):
    """Save the policy with the lowest cost on fixed validation windows."""

    def __init__(
        self,
        validation_prices,
        price_scale,
        model_path,
        eval_freq=25_000,
        n_validation_windows=100,
        window_len=24,
        seed=42,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        self.validation_prices = np.asarray(validation_prices, dtype=float)
        self.price_scale = float(price_scale)
        self.model_path = model_path
        self.eval_freq = max(int(eval_freq), 1)
        self.window_len = int(window_len)

        max_start = len(self.validation_prices) - self.window_len
        if max_start < 0:
            raise ValueError(
                "validation_prices must contain at least "
                f"{self.window_len} values."
            )
        if n_validation_windows <= 0:
            raise ValueError("n_validation_windows must be positive.")

        rng = np.random.default_rng(seed)
        n_available = max_start + 1
        n_selected = min(int(n_validation_windows), n_available)
        self.validation_starts = np.sort(
            rng.choice(n_available, size=n_selected, replace=False)
        )

        self.best_mean_cost = float("inf")
        self.best_step = None
        self.last_eval_step = None

    def _evaluate(self):
        costs = []
        total_rewards = []
        terminal_violations = []

        for start in self.validation_starts:
            prices_window = self.validation_prices[
                start : start + self.window_len
            ]
            _, powers, _, socs, rewards, prices_used = evaluate_model(
                self.model,
                prices_window,
                self.price_scale,
            )
            costs.append(compute_total_cost(prices_used, powers))
            total_rewards.append(float(np.sum(rewards)))
            terminal_violations.append(max(0.5 - float(socs[-1]), 0.0))

        mean_cost = float(np.mean(costs))
        mean_reward = float(np.mean(total_rewards))
        violation_rate = float(
            np.mean(np.asarray(terminal_violations) > 1e-6)
        )

        self.logger.record("validation/mean_cost", mean_cost)
        self.logger.record("validation/mean_reward", mean_reward)
        self.logger.record("validation/terminal_violation_rate", violation_rate)

        improved = mean_cost < self.best_mean_cost
        if improved:
            self.best_mean_cost = mean_cost
            self.best_step = int(self.num_timesteps)
            self.model.save(self.model_path)

        if self.verbose:
            status = "new best" if improved else "unchanged"
            print(
                f"Validation at step {self.num_timesteps}: "
                f"mean cost={mean_cost:.6f} EUR, "
                f"mean reward={mean_reward:.6f}, "
                f"terminal violation rate={violation_rate:.2%} "
                f"({status})"
            )

        self.last_eval_step = int(self.num_timesteps)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self):
        # Always consider the final policy, including short runs whose total
        # number of steps is smaller than eval_freq.
        if self.last_eval_step != self.num_timesteps:
            self._evaluate()


def train_model(
    prices_full,
    price_scale,
    log_dir="./logs",
    total_timesteps=100000,
    model_kwargs=None,
    algorithm=DEFAULT_ALGORITHM,
    model_path=None,
    validation_prices=None,
    eval_freq=25_000,
    n_validation_windows=100,
    validation_seed=42,
):
    """Train SAC or TD3 and return the best validation-cost checkpoint."""
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environment
    env_train = DummyVecEnv([lambda: make_train_env(prices_full, price_scale, log_dir)])

    algorithm = normalize_algorithm(algorithm)
    model_class = get_model_class(algorithm)
    model_path = model_path or get_model_path(algorithm)
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    # Default hyperparameters (override via model_kwargs)
    model_init_kwargs = {
        "buffer_size": 200000,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "tau": 0.005,
        "gamma": 1, # TODO: set gamma to 1(check result)
        "train_freq": 1,
        "gradient_steps": 1,
        "verbose": 1,
        "tensorboard_log": log_dir,
        # Terminal SOC penalties occur once per 24-step rollout.  Deliberately
        # include them in 25% of critic/actor batches once they are available.
        # "replay_buffer_class": TerminalStratifiedReplayBuffer,
        # "replay_buffer_kwargs": {"terminal_sample_fraction": 0.2},
    } # TODO: try 0 entropy
    if algorithm == "SAC":
        model_init_kwargs["policy_kwargs"] = {"net_arch": [256, 256, 256], "activation_fn": nn.ReLU}
    elif algorithm == "TD3":
        n_actions = env_train.action_space.shape[-1]
        model_init_kwargs.update(
            action_noise=NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.2 * np.ones(n_actions),
            ),
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            learning_starts=1000,
        )
    if model_kwargs:
        model_init_kwargs.update(model_kwargs)

    model = model_class("MlpPolicy", env_train, **model_init_kwargs)

    if validation_prices is None:
        validation_prices = prices_full
    validation_callback = BestValidationCostCallback(
        validation_prices=validation_prices,
        price_scale=price_scale,
        model_path=model_path,
        eval_freq=eval_freq,
        n_validation_windows=n_validation_windows,
        seed=validation_seed,
    )

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=validation_callback,
    )
    print(f"{algorithm} training complete!")

    # The callback saved the best policy. Reload it so callers cannot
    # accidentally evaluate the last in-memory policy instead.
    model = model_class.load(model_path, env=env_train)
    print(
        f"Best model from step {validation_callback.best_step} loaded from "
        f"{model_path}.zip (validation mean cost: "
        f"{validation_callback.best_mean_cost:.6f} EUR)"
    )

    return model


def tune_hyperparameters(
    prices_full, price_scale, log_dir="./logs", n_trials=50, total_timesteps=50000
):
    """Run Optuna study and return best hyperparameters."""
    from src.battery_model.hyperparameter_tuning import (
        tune_hyperparameters as run_tuning,
    )

    return run_tuning(
        prices_full,
        price_scale,
        log_dir=log_dir,
        n_trials=n_trials,
        total_timesteps=total_timesteps,
    )


def linear_schedule(initial_value: float):
    """Linear learning rate schedule that decays from initial_value to initial_value * 1e-1."""

    def func(progress_remaining: float) -> float:
        return np.max([progress_remaining * initial_value, initial_value * 1e-1])

    return func


def power_schedule(initial_value: float):
    """Power-law decay schedule using sqrt(progress_remaining)."""

    def func(progress_remaining: float) -> float:
        return initial_value * (progress_remaining**0.5)  # Slower decay

    return func


def run_multiple_training_evaluations(
    prices_full,
    prices_eval,
    global_price_scale,
    num_runs=32,
    base_lr=3e-4,
    total_timesteps=100000,
    algorithm=DEFAULT_ALGORITHM,
):
    """
    Train and evaluate the model with different LR schedules, and visualize results.

    Args:
        prices_full: Full electricity price data
        prices_eval: Evaluation price data window
        global_price_scale: Global price scaling factor
        num_runs: Number of training and evaluation runs (default: 32)
        base_lr: Base learning rate used by all schedules
        total_timesteps: Training timesteps per run
        algorithm: RL algorithm to train, "SAC" or "TD3"
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    algorithm = normalize_algorithm(algorithm)

    schedule_configs = {
        "constant": base_lr,
        "linear": linear_schedule(base_lr),
        "power": power_schedule(base_lr),
    }

    records = []
    total_costs_by_schedule = {name: [] for name in schedule_configs}

    print(f"\nTraining and evaluating {num_runs} {algorithm} runs per schedule...")
    for schedule_name, schedule_lr in schedule_configs.items():
        print(f"\n===== Schedule: {schedule_name} =====")
        for run in range(num_runs):
            print(f"\n--- {schedule_name} run {run + 1}/{num_runs} ---")

            model_kwargs = {
                "learning_rate": schedule_lr,
            }

            run_log_dir = f"./logs/{schedule_name}/run_{run}"
            model = train_model(
                prices_full,
                global_price_scale,
                total_timesteps=total_timesteps,
                log_dir=run_log_dir,
                model_kwargs=model_kwargs,
                algorithm=algorithm,
            )

            P_cmds, Ps, P_actuals, SOCs, rewards, prices_used = evaluate_model(
                model,
                prices_eval,
                global_price_scale,
            )

            cumulative_reward = float(np.sum(rewards))
            total_cost = compute_total_cost(prices_used, Ps)

            total_costs_by_schedule[schedule_name].append(total_cost)
            records.append(
                {
                    "schedule": schedule_name,
                    "run": run + 1,
                    "cumulative_reward": cumulative_reward,
                    "total_cost": total_cost,
                }
            )

            print(
                f"{schedule_name} run {run + 1} - "
                f"Cumulative reward: {cumulative_reward:.2f}, "
                f"Total cost: {total_cost:.2f} Euro"
            )

    # Build results dataframe
    results_df = pd.DataFrame(records)
    summary_df = results_df.groupby("schedule", as_index=False).agg(
        mean_cumulative_reward=("cumulative_reward", "mean"),
        std_cumulative_reward=("cumulative_reward", "std"),
        mean_total_cost=("total_cost", "mean"),
        std_total_cost=("total_cost", "std"),
    )
    cost_table_df = (
        results_df.pivot(index="run", columns="schedule", values="total_cost")
        .reset_index()
        .sort_values("run")
    )

    # Save raw and summary data
    results_filename = "rl_schedule_comparison_results.csv"
    summary_filename = "rl_schedule_comparison_summary.csv"
    cost_table_filename = "rl_schedule_comparison_total_cost_table.csv"
    results_df.to_csv(results_filename, index=False)
    summary_df.to_csv(summary_filename, index=False)
    cost_table_df.to_csv(cost_table_filename, index=False)

    # Create single combined box plot
    ordered_schedules = ["constant", "linear", "power"]
    plot_data = [total_costs_by_schedule[name] for name in ordered_schedules]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(plot_data)
    ax.set_xticks(range(1, len(ordered_schedules) + 1))
    ax.set_xticklabels([name.capitalize() for name in ordered_schedules])
    ax.set_ylabel("Total Cost (Euro)")
    ax.set_title(f"RL Total Cost over {num_runs} Runs")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_png_filename = "rl_schedule_comparison_boxplot.png"
    plot_pdf_filename = "rl_schedule_comparison_results.pdf"
    fig.savefig(plot_png_filename, dpi=300)

    # Save plot + summary + detailed total-cost table into one PDF
    with PdfPages(plot_pdf_filename) as pdf:
        pdf.savefig(fig)

        # Summary page
        fig_summary, ax_summary = plt.subplots(figsize=(8.27, 11.69))
        ax_summary.axis("off")
        ax_summary.set_title("Schedule Comparison Summary", fontsize=14, pad=16)
        summary_for_table = summary_df.round(4)
        summary_table = ax_summary.table(
            cellText=summary_for_table.values,
            colLabels=summary_for_table.columns,
            loc="center",
            cellLoc="center",
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(9)
        summary_table.scale(1.2, 1.5)
        pdf.savefig(fig_summary)
        plt.close(fig_summary)

        # Detailed total-cost table page(s)
        rows_per_page = 24
        cost_table_for_pdf = cost_table_df.round(4)
        for start_idx in range(0, len(cost_table_for_pdf), rows_per_page):
            chunk = cost_table_for_pdf.iloc[start_idx : start_idx + rows_per_page]
            fig_chunk, ax_chunk = plt.subplots(figsize=(8.27, 11.69))
            ax_chunk.axis("off")
            start_run = int(chunk["run"].iloc[0])
            end_run = int(chunk["run"].iloc[-1])
            ax_chunk.set_title(
                f"Total Cost by Run (runs {start_run}-{end_run})",
                fontsize=13,
                pad=16,
            )
            chunk_table = ax_chunk.table(
                cellText=chunk.values,
                colLabels=chunk.columns,
                loc="center",
                cellLoc="center",
            )
            chunk_table.auto_set_font_size(False)
            chunk_table.set_fontsize(9)
            chunk_table.scale(1.1, 1.4)
            pdf.savefig(fig_chunk)
            plt.close(fig_chunk)

    print(f"\nCombined box plot saved to {plot_png_filename}")
    print(f"Results PDF saved to {plot_pdf_filename}")
    print(f"Raw results saved to {results_filename}")
    print(f"Summary saved to {summary_filename}")
    print(f"Total-cost table saved to {cost_table_filename}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for _, row in summary_df.iterrows():
        print(
            f"{row['schedule']}: "
            f"Reward mean={row['mean_cumulative_reward']:.2f}, std={row['std_cumulative_reward']:.2f} | "
            f"Cost mean={row['mean_total_cost']:.2f}, std={row['std_total_cost']:.2f}"
        )

    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a Battery RL model with SAC or TD3")
    parser.add_argument(
        "--algorithm",
        type=normalize_algorithm,
        default=DEFAULT_ALGORITHM,
        choices=("SAC", "TD3"),
        help="RL algorithm to train",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning instead of a single training run",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run with --tune",
    )
    parser.add_argument(
        "--tune-timesteps",
        type=int,
        default=50000,
        help="Training timesteps per Optuna trial with --tune",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="TensorBoard/log directory",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Where to save the model without .zip",
    )
    parser.add_argument(
        "--use-price-model",
        action="store_true",
        help="Generate training prices with price_model instead of loading electricity_price.txt",
    )
    parser.add_argument(
        "--price-model-hours",
        type=int,
        default=DEFAULT_PRICE_MODEL_HOURS,
        help="Number of hourly prices to generate when --use-price-model is set",
    )
    parser.add_argument(
        "--price-model-seed",
        type=int,
        default=None,
        help="Random seed for price_model generation",
    )
    parser.add_argument(
        "--price-model-initial-price",
        type=float,
        default=None,
        help="Initial EUR/MWh price for price_model generation; defaults to price_seasonality(start_hour)",
    )
    parser.add_argument(
        "--price-model-start-hour",
        type=float,
        default=0.0,
        help="Starting hour used by price_model price seasonality",
    )
    parser.add_argument(
        "--price-model-dt-hours",
        type=float,
        default=1.0,
        help="Time step in hours for price_model generation",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Chronological fraction reserved for best-checkpoint selection",
    )
    parser.add_argument(
        "--validation-windows",
        type=int,
        default=100,
        help="Number of fixed validation windows used per checkpoint evaluation",
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
        help="Seed used to select the fixed validation windows",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load or generate price data
    if args.use_price_model:
        prices_full, global_price_scale = load_price_model_data(
            num_hours=args.price_model_hours,
            initial_price=args.price_model_initial_price,
            start_hour=args.price_model_start_hour,
            dt_hours=args.price_model_dt_hours,
            seed=args.price_model_seed,
        )
    else:
        prices_full, global_price_scale = load_price_data()

    if args.tune:
        tune_log_dir = args.log_dir or "./logs/hyperparameter_tuning"
        tune_hyperparameters(
            prices_full,
            global_price_scale,
            log_dir=tune_log_dir,
            n_trials=args.tune_trials,
            total_timesteps=args.tune_timesteps,
        )
        raise SystemExit(0)

    prices_train, prices_validation = split_train_validation_prices(
        prices_full,
        validation_fraction=args.validation_fraction,
    )
    print(
        f"Training prices: {len(prices_train)} hours | "
        f"Validation prices: {len(prices_validation)} hours"
    )
    prices_eval = prices_full[72:96]
    # optimize_battery_pyomo(prices_eval, verbose=False)

    # Optuna tuning (optional)
    # best_params = tune_hyperparameters(prices_full, global_price_scale,"./logs", n_trials=20, total_timesteps=50000)

    # Single run
    model_kwargs = {
        "learning_rate": linear_schedule(3e-4),
        "gamma": 0.99,
    }
    # model_kwargs = {
    #     'learning_rate': power_schedule(3e-4),
    # }
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = (
            "./logs/single_run_power_schedule"
            if args.algorithm == "SAC"
            else f"./logs/single_run_{args.algorithm.lower()}_power_schedule"
        )

    model = train_model(
        prices_train,
        global_price_scale,
        total_timesteps=args.timesteps,
        log_dir=log_dir,
        model_kwargs=model_kwargs,
        algorithm=args.algorithm,
        model_path=args.model_path,
        validation_prices=prices_validation,
        eval_freq=args.eval_freq,
        n_validation_windows=args.validation_windows,
        validation_seed=args.validation_seed,
    )

    # # Evaluate the model
    # P_cmds, Ps, P_actuals, SOCs, rewards, prices_used = evaluate_model(model, prices_eval, global_price_scale)

    # cumulative_reward = np.sum(rewards)
    # total_cost = compute_total_cost(prices_used, P_actuals)
    # print(f"Single Run - Cumulative reward: {cumulative_reward:.2f}, Total cost: {total_cost:.2f} Euro")

    # Train and evaluate multiple times
    # run_multiple_training_evaluations(prices_full, prices_eval, global_price_scale, num_runs=32, base_lr=3e-4)
