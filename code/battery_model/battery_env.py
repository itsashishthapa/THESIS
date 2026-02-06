"""
Battery Environment for Reinforcement Learning

- Observation: state-of-charge (SOC), current price, and normalized forecast window.
- Action: continuous charge/discharge command in [-1, 1] scaled to power [-P_max, P_max].
- Dynamics: `SOC_{t+1} = SOC_t + (eta * P_t * dt) / E_max` with bounds [0, 1].
- Reward: negative grid cost `- price_t * P_t / 1e6`.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BatteryEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, prices, P_max=20000.0, eta=0.9, E_max=2e5*3600.0, SOC_0=0.5, forecast_h=24, price_scale=None):
        super().__init__()
        self.prices = np.array(prices, dtype=np.float32)
        self.T = len(self.prices)
        self.dt = 3600.0  # in seconds
        self.P_max = float(P_max)
        self.eta = float(eta)
        self.E_max = float(E_max)
        self.SOC_0 = float(SOC_0)
        self.forecast_h = int(forecast_h)
        # Price scaling
        self.price_scale = float(price_scale) if price_scale is not None else float(np.percentile(np.abs(self.prices), 95) + 1e-6)
        # Observation: [SOC, normalized current price, normalized forecast window]
        # TODO: try adding the timestamp of the forcast window.
        obs_low = np.concatenate((np.array([0.0, -1.0], dtype=np.float32), -np.ones(self.forecast_h, dtype=np.float32)))
        obs_high = np.concatenate((np.array([1.0, 1.0], dtype=np.float32), np.ones(self.forecast_h, dtype=np.float32)))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # Action: continuous in [-1, 1], scaled to power [-P_max, P_max]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset(seed=42)

    def _forecast_window(self):
        idx = min(self.t, self.T - 1)
        window = self.prices[idx + 1: idx + 1 + self.forecast_h]
        if len(window) < self.forecast_h:
            pad_val = self.prices[-1]
            pad = np.full(self.forecast_h - len(window), pad_val, dtype=np.float32)
            window = np.concatenate([window, pad])
        return np.tanh(window / self.price_scale).astype(np.float32)

    def _obs(self):
        idx = min(self.t, self.T - 1)
        p = self.prices[idx]
        norm_price = float(np.tanh(p / self.price_scale))
        forecast = self._forecast_window()
        return np.concatenate((np.array([self.SOC, norm_price], dtype=np.float32), forecast))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.SOC = self.SOC_0
        self._last_P = 0.0
        return self._obs(), {}

    def step(self, action):
        # Scale action
        a = float(np.clip(action[0], -1.0, 1.0))
        P_cmd = a * self.P_max
        # Enforce SOC feasibility: compute max allowed charge/discharge power for this step
        E_room_charge = (1.0 - self.SOC) * self.E_max
        E_room_discharge = self.SOC * self.E_max
        max_charge_P = min(self.P_max, (E_room_charge / self.dt) / max(self.eta, 1e-6))
        max_discharge_P = min(self.P_max, (E_room_discharge / self.dt) / max(self.eta, 1e-6))
        if P_cmd >= 0:
            P = float(np.clip(P_cmd, 0.0, max_charge_P))
        else:
            P = float(np.clip(P_cmd, -max_discharge_P, 0.0))
        P_actual = self.eta * P
        dSOC = (P_actual * self.dt) / self.E_max
        self.SOC = float(np.clip(self.SOC + dSOC, 0.0, 1.0))
        price = float(self.prices[self.t])
        cost = (price / 1e6) * P_actual
        reward = float(-cost)
        self._last_P = P
        self.t += 1
        # TODO: investigate on final SOC and add a terminal reward/penalty if needed.
        terminated = self.t >= self.T
        truncated = False
        info = {'P': P, 'P_actual': P_actual, 'price': price, 'SOC': self.SOC, 'raw_reward': reward}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        print(f't={self.t} SOC={self.SOC:.3f} P={self._last_P:.1f}')


class RandomWindowEnv(gym.Env):
    """Wrapper that samples random windows from full price data on each reset"""
    def __init__(self, prices_full, window_len=24, price_scale=None, **kwargs):
        self.prices_full = prices_full
        self.window_len = window_len
        self.max_start = len(prices_full) - window_len
        self.kwargs = kwargs
        self.price_scale = price_scale
        # Create a dummy env to get spaces
        dummy = BatteryEnv(prices_full[:window_len], price_scale=self.price_scale, **kwargs)
        self.observation_space = dummy.observation_space
        self.action_space = dummy.action_space
        self.metadata = dummy.metadata
        self._sample_new_window()
        
    def _sample_new_window(self):
        start = np.random.randint(0, max(1, self.max_start))
        window_prices = self.prices_full[start:start + self.window_len]
        self.env = BatteryEnv(window_prices, price_scale=self.price_scale, **self.kwargs)
    
    def reset(self, **kwargs):
        self._sample_new_window()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
