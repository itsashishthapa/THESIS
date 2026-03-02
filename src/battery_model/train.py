"""
Training script for Battery RL model using SAC (Soft Actor-Critic)
"""

import logging
import os
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# from Battery_model.Optimization_Pyomo.model_pyomo import optimize_battery_pyomo
from src.battery_model.battery_env import RandomWindowEnv
from src.battery_model.evaluate import evaluate_model


def setup_optuna_logging(log_dir):
    """Setup logging for Optuna optimization process"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'optuna_optimization_{timestamp}.log')
    
    logger = logging.getLogger('optuna_logger')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    logger.addHandler(file_handler)
    
    return logger, log_file


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
        gamma=0.99,
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
    model.save("battery_sac_model")
    print("Model saved to battery_sac_model.zip")
    
    return model


def _optuna_objective(trial, prices_full, price_scale, log_dir, total_timesteps=50000, logger=None):
    """Optuna objective for SAC hyperparameter tuning."""
    trial_log_dir = os.path.join(log_dir, f'optuna_trial_{trial.number}')
    os.makedirs(trial_log_dir, exist_ok=True)

    try:
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
        update_ratio = trial.suggest_categorical("update_ratio", [0.25, 0.5, 1.0, 2.0])
        gradient_steps = max(1, int(train_freq * update_ratio))

        model_kwargs = {
            'buffer_size': trial.suggest_categorical('buffer_size', [100000, 200000, 500000]),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'tau': trial.suggest_float('tau', 0.001, 0.05, log=True),
            'gamma': trial.suggest_float('gamma', 0.95, 0.9999, log=True),
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            'ent_coef': trial.suggest_categorical("ent_coef",["auto", "auto_0.1", 0.003, 0.01, 0.03]),
            'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
            'tensorboard_log': trial_log_dir,
            'verbose': 0,
        }

        if logger:
            logger.info(f"Starting trial {trial.number} with params: {model_kwargs}")

        env_train = DummyVecEnv([lambda: make_train_env(prices_full, price_scale, trial_log_dir)])
        model = SAC('MlpPolicy', env_train, **model_kwargs)
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
        # Return a poor reward to penalize failed trials rather than crashing
        return float('-inf')


def tune_hyperparameters(prices_full, price_scale, log_dir='./logs', n_trials=50, total_timesteps=50000):
    """Run Optuna study and return best hyperparameters."""
    logger, log_file = setup_optuna_logging(log_dir)
    
    logger.info(f"Starting Optuna hyperparameter optimization with {n_trials} trials")
    logger.info(f"Total timesteps per trial: {total_timesteps}")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: _optuna_objective(trial, prices_full, price_scale, log_dir, total_timesteps, logger),
        n_trials=n_trials,
    )
    
    logger.info(f'Best reward: {study.best_value}')
    logger.info(f'Best params: {study.best_params}')
    logger.info(f'Optimization completed. Log file: {log_file}')
    
    print('Best reward:', study.best_value)
    print('Best params:', study.best_params)
    return study.best_params


def run_multiple_training_evaluations(prices_full, prices_eval, global_price_scale, num_runs=32):
    """
    Train and evaluate the model multiple times, and visualize results.
    
    Args:
        prices_full: Full electricity price data
        prices_eval: Evaluation price data window
        global_price_scale: Global price scaling factor
        num_runs: Number of training and evaluation runs (default: 32)
    """
    import matplotlib.pyplot as plt
    
    all_cumulative_rewards = []
    all_total_costs = []
    
    print(f"\nTraining and evaluating {num_runs} times...")
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        # Train the model
        model = train_model(prices_full, global_price_scale, total_timesteps=50000, log_dir=f'./logs/run_{run}')
        
        # Evaluate the model
        P_cmds, Ps, P_actuals, SOCs, rewards, prices_used = evaluate_model(model, prices_eval, global_price_scale)
        
        # Store metrics
        cumulative_reward = np.sum(rewards)
        total_cost = np.sum((prices_used / 1e6) * P_actuals) 
        
        all_cumulative_rewards.append(cumulative_reward)
        all_total_costs.append(total_cost)
        
        print(f"Run {run + 1} - Cumulative reward: {cumulative_reward:.2f}, Total cost: {total_cost:.2f} Euro")
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.boxplot(all_total_costs)
    ax.set_ylabel('Total Cost (Euro)')
    ax.set_title(f'RL Total Cost over {num_runs} Runs')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_boxplot_results.png', dpi=300)
    print("\nBox plot saved to rl_boxplot_results.png")
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Cumulative Reward - Mean: {np.mean(all_cumulative_rewards):.2f}, Std: {np.std(all_cumulative_rewards):.2f}")
    print(f"Total Cost - Mean: {np.mean(all_total_costs):.2f}, Std: {np.std(all_total_costs):.2f}")


if __name__ == '__main__':
    # Load price data
    prices_full, global_price_scale = load_price_data()
    prices_eval = prices_full[72:96]
    # optimize_battery_pyomo(prices_eval, verbose=False)

    # Optuna tuning (optional)
    # best_params = tune_hyperparameters(prices_full, global_price_scale,"./logs", n_trials=20, total_timesteps=50000)

    # Train and evaluate multiple times
    run_multiple_training_evaluations(prices_full, prices_eval, global_price_scale, num_runs=32)
    
