"""
Training script for Battery RL model using SAC (Soft Actor-Critic)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from battery_env import RandomWindowEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results
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


def train_model(prices_full, price_scale, log_dir='./logs', total_timesteps=100000):
    """Train the SAC model"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized environment
    env_train = DummyVecEnv([lambda: make_train_env(prices_full, price_scale, log_dir)])

    # TODO: perform hyperparameter tuning
    model = SAC('MlpPolicy', env_train,
                buffer_size=200000,
                batch_size=256,
                learning_rate=3e-4,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                verbose=1,
                tensorboard_log=log_dir)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    print("SAC training complete!")
    
    # Save the model
    model.save("battery_sac_model")
    print("Model saved to battery_sac_model.zip")
    
    return model


if __name__ == '__main__':
    # Load price data
    prices_full, global_price_scale = load_price_data()
    
    # Train the model
    model = train_model(prices_full, global_price_scale, total_timesteps=200000)
    
