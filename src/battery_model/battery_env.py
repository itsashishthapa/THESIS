"""
Battery Environment for Reinforcement Learning

- Observation: state-of-charge (SOC), current price, normalized time, and normalized forecast window.
- Action: continuous charge/discharge command in [-1, 1] mapped smoothly to power.
- Dynamics: `SOC_{t+1} = SOC_t + (eta * P_t * dt) / E_max` with bounds [0, 1].
- Reward: negative grid cost `- price_t * P_t / 1e6`.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BatteryEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        prices,
        P_max=20000.0,
        eta=0.9,
        E_max=2e5*3600.0,
        SOC_0=0.5,
        forecast_h=24,
        price_scale=None,
        smooth_k=500.0,
        deadband_eps=0.0025,
    ):
        super().__init__()
        self.prices = np.array(prices, dtype=np.float32)
        self.T = len(self.prices)
        self.dt = 3600.0  # in seconds
        self.P_max = float(P_max)
        self.eta = float(eta)
        self.E_max = float(E_max)
        self.SOC_0 = float(SOC_0)
        self.forecast_h = int(forecast_h)
        self.smooth_k = float(smooth_k)
        self.deadband_eps = float(deadband_eps)
        self.soc_upper_gate = 0.995
        self.soc_lower_gate = 0.005
        # Price scaling
        self.price_scale = float(price_scale) if price_scale is not None else float(np.percentile(np.abs(self.prices), 95) + 1e-6)
        # Observation: [SOC, normalized current price, normalized time, normalized forecast window]
        obs_low = np.concatenate((np.array([0.0, -1.0, 0.0], dtype=np.float32), -np.ones(self.forecast_h, dtype=np.float32)))
        obs_high = np.concatenate((np.array([1.0, 1.0, 1.0], dtype=np.float32), np.ones(self.forecast_h, dtype=np.float32)))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # Action: continuous in [-1, 1], smoothly mapped to power.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset(seed=42)

    def _smooth_gate(self, x):
        return 0.5 * (1.0 + np.tanh(self.smooth_k * x))

    def _smooth_power_mapping(self, action_value):
        u = float(action_value)
        if not np.isfinite(u):
            raise ValueError("BatteryEnv action must be finite.")
        if u < -1.0 or u > 1.0:
            raise ValueError("BatteryEnv action must be within [-1, 1].")
        charge_action_gate = self._smooth_gate(u - self.deadband_eps)
        discharge_action_gate = 1.0 - self._smooth_gate(u + self.deadband_eps)
        charge_soc_gate = self._smooth_gate(self.soc_upper_gate - self.SOC)
        discharge_soc_gate = self._smooth_gate(self.SOC - self.soc_lower_gate)
        P = self.P_max * (
            charge_action_gate * u * charge_soc_gate
            + discharge_action_gate * u * discharge_soc_gate
        )
        return float(P), u

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
        denom = max(self.T - 1, 1)
        norm_time = float(min(self.t, self.T - 1) / denom)
        forecast = self._forecast_window()
        return np.concatenate((np.array([self.SOC, norm_price, norm_time], dtype=np.float32), forecast))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.SOC = self.SOC_0
        self._last_P = 0.0
        return self._obs(), {}

    def step(self, action):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        P, a = self._smooth_power_mapping(action_value)
        P_cmd = a * self.P_max
        P_actual = self.eta * P
        dSOC = (P_actual * self.dt) / self.E_max
        self.SOC = float(np.clip(self.SOC + dSOC, 0.0, 1.0))
        price = float(self.prices[self.t])
        cost = (price / 1e6) * P
        reward = float(-cost)
        self._last_P = P
        self.t += 1
        # TODO: investigate on final SOC and add a terminal reward/penalty if needed.
        terminated = self.t >= self.T
        truncated = False
        info = {
            'P': P,
            'P_cmd': P_cmd,
            'P_actual': P_actual,
            'u': a,
            'price': price,
            'SOC': self.SOC,
            'cost': cost,
            'raw_reward': reward,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        print(f't={self.t} SOC={self.SOC:.3f} P={self._last_P:.1f}')


class RandomWindowEnv(gym.Env):
    """Wrapper that samples random windows from full price data on each reset"""
    def __init__(self, prices_full, window_len=24, price_scale=None, **kwargs):
        self.prices_full = prices_full
        self.window_len = window_len
        if window_len > len(prices_full):
            raise ValueError("window_len cannot be larger than the available price history.")
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
        start = np.random.randint(0, self.max_start + 1)
        window_prices = self.prices_full[start:start + self.window_len]
        self.env = BatteryEnv(window_prices, price_scale=self.price_scale, **self.kwargs)
    
    def reset(self, **kwargs):
        self._sample_new_window()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
