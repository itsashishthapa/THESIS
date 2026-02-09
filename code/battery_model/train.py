"""
Training script for Battery RL model using SAC (Soft Actor-Critic)
"""

import os

import numpy as np
import optuna
import pandas as pd
from battery_env import RandomWindowEnv
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def load_price_data():
    """Load electricity price data from file"""
    price_df = pd.read_csv('../../Battery_model/Optimization_Pyomo/input_data/electricity_price.txt',
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


def _optuna_objective(trial, prices_full, price_scale, log_dir, total_timesteps=50000):
    """Optuna objective for SAC hyperparameter tuning."""
    trial_log_dir = os.path.join(log_dir, f'optuna_trial_{trial.number}')
    os.makedirs(trial_log_dir, exist_ok=True)

    model_kwargs = {
        'buffer_size': trial.suggest_categorical('buffer_size', [100000, 200000, 500000]),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 3e-4, log=True),
        'tau': trial.suggest_float('tau', 0.001, 0.02, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.9999, log=True),
        'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
        'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 4, 8]),
        'tensorboard_log': trial_log_dir,
        'verbose': 0,
    }

    env_train = DummyVecEnv([lambda: make_train_env(prices_full, price_scale, trial_log_dir)])
    model = SAC('MlpPolicy', env_train, **model_kwargs)
    model.learn(total_timesteps=total_timesteps)

    mean_reward, _ = evaluate_policy(model, env_train, n_eval_episodes=5, deterministic=True)
    env_train.close()
    return mean_reward


def tune_hyperparameters(prices_full, price_scale, log_dir='./logs', n_trials=20, total_timesteps=50000):
    """Run Optuna study and return best hyperparameters."""
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: _optuna_objective(trial, prices_full, price_scale, log_dir, total_timesteps),
        n_trials=n_trials,
    )
    print('Best reward:', study.best_value)
    print('Best params:', study.best_params)
    return study.best_params


if __name__ == '__main__':
    # Load price data
    prices_full, global_price_scale = load_price_data()

    # Optuna tuning (optional)
    best_params = tune_hyperparameters(prices_full, global_price_scale, n_trials=20, total_timesteps=50000)

    # Train the model with best params
    model = train_model(prices_full, global_price_scale, total_timesteps=100000, model_kwargs=best_params)
    
