"""
Evaluation script for trained Battery RL model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from src.battery_model.battery_env import BatteryEnv

DEFAULT_MODEL_PATH = "src/battery_model/battery_sac_model"


def load_price_data():
    """Load electricity price data from file"""
    price_df = pd.read_csv('Battery_model/Optimization_Pyomo/input_data/electricity_price.txt',
                            sep=r'\s+', header=None, names=['t', 'price'])
    prices_full = price_df['price'].values.astype(float)
    
    # Global scaler (95th percentile of absolute prices across full dataset)
    global_price_scale = np.percentile(np.abs(prices_full), 95) + 1e-6
    
    print(f'Full dataset length: {len(prices_full)} hours')
    print(f'Global price scale (95th pct): {global_price_scale:.4f}')
    
    return prices_full, global_price_scale


def compute_total_cost(prices, power):
    """Compute total grid cost in Euro for the given electricity prices and power values."""
    return float(np.sum((np.asarray(prices) / 1e6) * np.asarray(power)))


def evaluate_model(model, prices_eval, price_scale):
    """Evaluate the trained model on given price data"""
    env_eval = BatteryEnv(prices_eval, price_scale=price_scale)
    model_obs_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if model_obs_shape is not None and env_eval.observation_space.shape != model_obs_shape:
        raise ValueError(
            "Model observation shape "
            f"{model_obs_shape} does not match BatteryEnv shape "
            f"{env_eval.observation_space.shape}. Use the same BatteryEnv "
            "observation features and forecast_h as training, or retrain the model."
        )
    obs, _ = env_eval.reset()
    
    P_cmds, Ps, P_actuals, SOCs, rewards, prices_used = [], [], [], [], [], []
    
    for _ in range(env_eval.T):
        action, _ = model.predict(obs, deterministic=True)
        # commanded power before env clipping
        P_cmd = float(np.clip(action[0], -1.0, 1.0)) * env_eval.P_max
        obs, reward, terminated, truncated, info = env_eval.step(action)
        
        P_cmds.append(P_cmd)
        Ps.append(info['P'])  # P before eta
        P_actuals.append(info.get('P_actual', info['P']))  # P_actual used in reward calc (post-eta)
        SOCs.append(info['SOC'])
        rewards.append(reward)
        prices_used.append(info['price'])
        
        if terminated or truncated:
            break
    
    # Convert to numpy arrays
    P_cmds = np.array(P_cmds)
    Ps = np.array(Ps)
    P_actuals = np.array(P_actuals)
    SOCs = np.array(SOCs)
    rewards = np.array(rewards)
    prices_used = np.array(prices_used)
    
    return P_cmds, Ps, P_actuals, SOCs, rewards, prices_used


def plot_evaluation_results(P_cmds, Ps, P_actuals, SOCs, rewards, prices_used):
    """Plot evaluation results"""
    cum_cost_preeta = compute_total_cost(prices_used, Ps)
    cum_cost_posteta = compute_total_cost(prices_used, P_actuals)
    
    print(f'RL Cumulative cost (pre-eta, Euro): {cum_cost_preeta:.2f}')
    print(f'RL Cumulative cost (post-eta, Euro): {cum_cost_posteta:.2f}')
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Plot prices
    axs[0].plot(prices_used)
    axs[0].set_ylabel('Price')
    axs[0].set_title('RL Policy')
    axs[0].grid(True)
    
    # Plot power
    axs[1].plot(P_cmds, label='P_cmd')
    axs[1].plot(Ps, label='P (pre-eta)')
    axs[1].plot(P_actuals, label='P_actual')
    axs[1].set_ylabel('P [W]')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot SOC
    axs[2].plot(SOCs)
    axs[2].set_ylabel('SOC')
    axs[2].grid(True)
    
    # Plot cumulative reward
    axs[3].plot(np.cumsum(rewards))
    axs[3].set_ylabel('Cumulative reward')
    axs[3].set_xlabel('Time step (h)')
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load price data
    prices_full, global_price_scale = load_price_data()
    
    # Load trained model
    model = SAC.load(DEFAULT_MODEL_PATH)
    print("Model loaded successfully!")
    
    # Evaluate
    prices_eval = prices_full[72:96]
    print(f'Evaluation window: {len(prices_eval)} hours')
    
    # Run evaluation
    P_cmds, Ps, P_actuals, SOCs, rewards, prices_used = evaluate_model(
        model, prices_eval, global_price_scale
    )
    
    # Plot results
    plot_evaluation_results(P_cmds, Ps, P_actuals, SOCs, rewards, prices_used)
    
