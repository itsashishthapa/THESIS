"""
Battery Environment for Reinforcement Learning

- Observation: state-of-charge (SOC), current price, normalized time, and normalized forecast window.
- Action: continuous charge/discharge command in [-1, 1] mapped smoothly to power.
- Dynamics: `SOC_{t+1} = SOC_t + (P_actual_t * dt) / E_max` with bounds [0, 1].
- Reward: negative grid-side cost `- price_t * P_t * dt`.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BatteryEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        prices,
        P_max=20.0,
        eta=0.9,
        E_max=2e2,
        SOC_0=0.5,
        forecast_h=24,
        price_scale=None,
        smooth_k=700.0,
        deadband_eps=0.0025,
        efficiency_eps=1.0,
        terminal_soc_min=0.5,
        terminal_soc_penalty=10.0,
        dt=1.0,
        price_unit_scale=1000.0,
    ):
        super().__init__()
        self.prices = np.array(prices, dtype=np.float32)
        self.T = len(self.prices)
        self.dt = float(dt)  # v2 uses hours with kW/kWh-style units.
        self.P_max = float(P_max)
        self.eta = float(eta)
        self.E_max = float(E_max)
        self.SOC_0 = float(SOC_0)
        self.forecast_h = int(forecast_h)
        self.smooth_k = float(smooth_k)
        self.deadband_eps = float(deadband_eps)
        self.efficiency_eps = float(efficiency_eps)
        self.terminal_soc_min = float(terminal_soc_min)
        self.terminal_soc_penalty = float(terminal_soc_penalty)
        self.price_unit_scale = float(price_unit_scale)
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

    def _actual_power_mapping(self, P):
        arg = P / self.efficiency_eps
        tanh_arg = np.tanh(arg)
        efficiency_factor = 0.5 * (
            (1.0 + tanh_arg) * self.eta
            + (1.0 - tanh_arg) / self.eta
        )
        P_actual = P * efficiency_factor
        return float(P_actual), float(arg), float(efficiency_factor)

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
        self._last_P_actual = 0.0
        self.acc_cost = 0.0
        return self._obs(), {}

    def step(self, action):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        P, a = self._smooth_power_mapping(action_value)
        P_cmd = a * self.P_max
        P_actual, arg, efficiency_factor = self._actual_power_mapping(P)
        dSOC = (P_actual * self.dt) / self.E_max
        self.SOC = float(np.clip(self.SOC + dSOC, 0.0, 1.0))
        price = float(self.prices[self.t])
        cost = (price / self.price_unit_scale) * P * self.dt
        self.acc_cost += cost
        grid_reward = float(-cost)
        reward = grid_reward
        self._last_P = P
        self._last_P_actual = P_actual
        self.t += 1
        terminated = self.t >= self.T
        terminal_soc_violation = 0.0
        terminal_penalty = 0.0
        if terminated:
            terminal_soc_violation = max(self.terminal_soc_min - self.SOC, 0.0)
            terminal_penalty = self.terminal_soc_penalty * terminal_soc_violation
            reward -= terminal_penalty
        truncated = False
        info = {
            'P': P,
            'P_cmd': P_cmd,
            'P_actual': P_actual,
            'arg': arg,
            'efficiency_factor': efficiency_factor,
            'u': a,
            'price': price,
            'SOC': self.SOC,
            'cost': cost,
            'acc_cost': self.acc_cost,
            'terminal_soc_min': self.terminal_soc_min,
            'terminal_soc_violation': terminal_soc_violation,
            'terminal_penalty': terminal_penalty,
            'grid_reward': grid_reward,
            'raw_reward': reward,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        print(f't={self.t} SOC={self.SOC:.3f} P={self._last_P:.1f} P_actual={self._last_P_actual:.1f}')


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
