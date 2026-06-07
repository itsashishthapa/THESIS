"""
Training script for Battery RL model using SAC (Soft Actor-Critic)
"""

import os

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# from Battery_model.Optimization_Pyomo.model_pyomo import optimize_battery_pyomo
from src.battery_model.battery_env import RandomWindowEnv
from src.battery_model.evaluate import (
    DEFAULT_MODEL_PATH,
    compute_total_cost,
    evaluate_model,
)


def load_price_data():
    """Load electricity price data from file"""
    price_df = pd.read_csv('./Battery_model/Optimization_Pyomo/input_data/electricity_price.txt',
                            sep=r'\s+', header=None, names=['t', 'price'])
    prices_full = price_df['price'].values.astype(float)
    
    # Global scaler (95th percentile of absolute prices across full dataset)
    global_price_scale = np.percentile(np.abs(prices_full), 95) + 1e-6
    
    print(f'Full dataset length: {len(prices_full)} hours')
    print(f'Global price scale (95th pct): {global_price_scale:.4f}')
    
    return prices_full, global_price_scale


def make_train_env(prices_full, price_scale, log_dir):
    """Create a training environment with random price windows"""
    env = RandomWindowEnv(prices_full, window_len=24, price_scale=price_scale)
    env = Monitor(env)  
    return env


def train_model(prices_full, price_scale, log_dir='./logs', total_timesteps=100000, model_kwargs=None):
    """Train the SAC model"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized environment
    env_train = DummyVecEnv([lambda: make_train_env(prices_full, price_scale, log_dir)])

    # Default hyperparameters (override via model_kwargs)
    sac_kwargs = dict(
        buffer_size=200000,
        batch_size=256,
        learning_rate=3e-4,
        tau=0.005,
        gamma=0.9999,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_dir,
    )
    if model_kwargs:
        sac_kwargs.update(model_kwargs)

    model = SAC('MlpPolicy', env_train, **sac_kwargs)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    print("SAC training complete!")
    
    # Save the model
    os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
    model.save(DEFAULT_MODEL_PATH)
    print(f"Model saved to {DEFAULT_MODEL_PATH}.zip")
    
    return model


def tune_hyperparameters(prices_full, price_scale, log_dir='./logs', n_trials=50, total_timesteps=50000):
    """Run Optuna study and return best hyperparameters."""
    from src.battery_model.hyperparameter_tuning import tune_hyperparameters as run_tuning

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
        return np.max([progress_remaining * initial_value,
                       initial_value * 1e-1])
    return func

def power_schedule(initial_value: float):
    """Power-law decay schedule using sqrt(progress_remaining)."""
    def func(progress_remaining: float) -> float:
        return initial_value * (progress_remaining ** 0.5)  # Slower decay
    return func


def run_multiple_training_evaluations(
    prices_full,
    prices_eval,
    global_price_scale,
    num_runs=32,
    base_lr=3e-4,
    total_timesteps=100000,
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
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    schedule_configs = {
        'constant': base_lr,
        'linear': linear_schedule(base_lr),
        'power': power_schedule(base_lr),
    }

    records = []
    total_costs_by_schedule = {name: [] for name in schedule_configs}

    print(f"\nTraining and evaluating {num_runs} runs per schedule...")
    for schedule_name, schedule_lr in schedule_configs.items():
        print(f"\n===== Schedule: {schedule_name} =====")
        for run in range(num_runs):
            print(f"\n--- {schedule_name} run {run + 1}/{num_runs} ---")

            model_kwargs = {
                'learning_rate': schedule_lr,
            }

            run_log_dir = f'./logs/{schedule_name}/run_{run}'
            model = train_model(
                prices_full,
                global_price_scale,
                total_timesteps=total_timesteps,
                log_dir=run_log_dir,
                model_kwargs=model_kwargs,
            )

            P_cmds, Ps, P_actuals, SOCs, rewards, prices_used = evaluate_model(
                model,
                prices_eval,
                global_price_scale,
            )

            cumulative_reward = float(np.sum(rewards))
            total_cost = compute_total_cost(prices_used, Ps)

            total_costs_by_schedule[schedule_name].append(total_cost)
            records.append({
                'schedule': schedule_name,
                'run': run + 1,
                'cumulative_reward': cumulative_reward,
                'total_cost': total_cost,
            })

            print(
                f"{schedule_name} run {run + 1} - "
                f"Cumulative reward: {cumulative_reward:.2f}, "
                f"Total cost: {total_cost:.2f} Euro"
            )

    # Build results dataframe
    results_df = pd.DataFrame(records)
    summary_df = (
        results_df
        .groupby('schedule', as_index=False)
        .agg(
            mean_cumulative_reward=('cumulative_reward', 'mean'),
            std_cumulative_reward=('cumulative_reward', 'std'),
            mean_total_cost=('total_cost', 'mean'),
            std_total_cost=('total_cost', 'std'),
        )
    )
    cost_table_df = (
        results_df
        .pivot(index='run', columns='schedule', values='total_cost')
        .reset_index()
        .sort_values('run')
    )

    # Save raw and summary data
    results_filename = 'rl_schedule_comparison_results.csv'
    summary_filename = 'rl_schedule_comparison_summary.csv'
    cost_table_filename = 'rl_schedule_comparison_total_cost_table.csv'
    results_df.to_csv(results_filename, index=False)
    summary_df.to_csv(summary_filename, index=False)
    cost_table_df.to_csv(cost_table_filename, index=False)

    # Create single combined box plot
    ordered_schedules = ['constant', 'linear', 'power']
    plot_data = [total_costs_by_schedule[name] for name in ordered_schedules]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(plot_data)
    ax.set_xticks(range(1, len(ordered_schedules) + 1))
    ax.set_xticklabels([name.capitalize() for name in ordered_schedules])
    ax.set_ylabel('Total Cost (Euro)')
    ax.set_title(f'RL Total Cost over {num_runs} Runs')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_png_filename = 'rl_schedule_comparison_boxplot.png'
    plot_pdf_filename = 'rl_schedule_comparison_results.pdf'
    fig.savefig(plot_png_filename, dpi=300)

    # Save plot + summary + detailed total-cost table into one PDF
    with PdfPages(plot_pdf_filename) as pdf:
        pdf.savefig(fig)

        # Summary page
        fig_summary, ax_summary = plt.subplots(figsize=(8.27, 11.69))
        ax_summary.axis('off')
        ax_summary.set_title('Schedule Comparison Summary', fontsize=14, pad=16)
        summary_for_table = summary_df.round(4)
        summary_table = ax_summary.table(
            cellText=summary_for_table.values,
            colLabels=summary_for_table.columns,
            loc='center',
            cellLoc='center',
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
            chunk = cost_table_for_pdf.iloc[start_idx:start_idx + rows_per_page]
            fig_chunk, ax_chunk = plt.subplots(figsize=(8.27, 11.69))
            ax_chunk.axis('off')
            start_run = int(chunk['run'].iloc[0])
            end_run = int(chunk['run'].iloc[-1])
            ax_chunk.set_title(
                f'Total Cost by Run (runs {start_run}-{end_run})',
                fontsize=13,
                pad=16,
            )
            chunk_table = ax_chunk.table(
                cellText=chunk.values,
                colLabels=chunk.columns,
                loc='center',
                cellLoc='center',
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


if __name__ == '__main__':
    # Load price data
    prices_full, global_price_scale = load_price_data()
    prices_eval = prices_full[72:96]
    # optimize_battery_pyomo(prices_eval, verbose=False)

    # Optuna tuning (optional)
    # best_params = tune_hyperparameters(prices_full, global_price_scale,"./logs", n_trials=20, total_timesteps=50000)

    # Single run 
    model_kwargs = {
        'learning_rate': linear_schedule(3e-4),
        'gamma': 0.9999,
    }
    # model_kwargs = {
    #     'learning_rate': power_schedule(3e-4),
    # }
    model = train_model(prices_full, global_price_scale, total_timesteps=500000, 
                       log_dir='./logs/single_run_power_schedule', model_kwargs=model_kwargs)
    
    # # Evaluate the model
    # P_cmds, Ps, P_actuals, SOCs, rewards, prices_used = evaluate_model(model, prices_eval, global_price_scale)
    
    # cumulative_reward = np.sum(rewards)
    # total_cost = np.sum((prices_used / 1e6) * P_actuals)
    # print(f"Single Run - Cumulative reward: {cumulative_reward:.2f}, Total cost: {total_cost:.2f} Euro")
    
    # Train and evaluate multiple times
    # run_multiple_training_evaluations(prices_full, prices_eval, global_price_scale, num_runs=32, base_lr=3e-4)
    
