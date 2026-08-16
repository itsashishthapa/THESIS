"""
Battery Environment for Reinforcement Learning

Z- Observation: state-of-charge (SOC), current price, normalized time, and the full normalized forecast.
- Action: continuous charge/discharge command in [-1, 1] mapped smoothly to power.
- Dynamics: `SOC_{t+1} = SOC_t + (P_actual_t * dt) / E_max` with bounds [0, 1].
- Feasibility: realized power is limited to the energy that can be discharged
  from or stored in the battery during the current time step.
- Reward: negative grid-side cost `- price_t * P_t * dt`.
- Reachability: falling below the SOC needed to retain a feasible path to the
  terminal target receives a progressive penalty.
- Terminal SOC: ending below `terminal_soc_min` receives a penalty.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BatteryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}  # noqa: RUF012

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
        terminal_soc_reachability_penalty=10.0,
        terminal_soc_penalty=40.0,
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
        self.terminal_soc_reachability_penalty = float(
            terminal_soc_reachability_penalty
        )
        self.terminal_soc_penalty = float(terminal_soc_penalty)
        self.price_unit_scale = float(price_unit_scale)
        self.soc_upper_gate = 0.995
        self.soc_lower_gate = 0.005
        # Price scaling
        self.price_scale = (
            float(price_scale)
            if price_scale is not None
            else float(np.percentile(np.abs(self.prices), 95) + 1e-6)
        )
        # The same complete forecast is included at every time step.
        # With the standard setup, forecast_h equals the 24-step episode length.
        obs_low = np.concatenate(
            (
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
                -np.ones(self.forecast_h, dtype=np.float32),
            )
        )
        obs_high = np.concatenate(
            (
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                np.ones(self.forecast_h, dtype=np.float32),
            )
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        # Action: continuous in [-1, 1], smoothly mapped to power.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset(seed=42)

    def _smooth_gate(self, x):
        return 0.5 * (1.0 + np.tanh(self.smooth_k * x))

    def _smooth_power_mapping(self, action_value):
        u = float(action_value)
        charge_action_gate = self._smooth_gate(u - self.deadband_eps)
        discharge_action_gate = 1.0 - self._smooth_gate(u + self.deadband_eps)
        charge_soc_gate = self._smooth_gate(self.soc_upper_gate - self.SOC)
        discharge_soc_gate = self._smooth_gate(self.SOC - self.soc_lower_gate)
        P = self.P_max * (
            charge_action_gate * u * charge_soc_gate
            + self.eta * discharge_action_gate * u * discharge_soc_gate
        )
        return float(P), u

    def _actual_power_mapping(self, P):
        arg = P / self.efficiency_eps
        tanh_arg = np.tanh(arg)
        efficiency_factor = 0.5 * (
            (1.0 + tanh_arg) * self.eta + (1.0 - tanh_arg) / self.eta
        )
        P_actual = P * efficiency_factor
        return float(P_actual), float(arg), float(efficiency_factor)

    def _grid_power_for_actual_power(self, P_actual_target, P_reference):
        """Invert the smooth efficiency mapping within the requested power range."""
        if P_actual_target == 0.0:
            return 0.0

        lower = min(0.0, float(P_reference))
        upper = max(0.0, float(P_reference))
        for _ in range(32):
            midpoint = 0.5 * (lower + upper)
            midpoint_actual, _, _ = self._actual_power_mapping(midpoint)
            if midpoint_actual < P_actual_target:
                lower = midpoint
            else:
                upper = midpoint
        return float(0.5 * (lower + upper))

    def _limit_power_to_available_energy(self, P, P_actual):
        """Project requested power onto the one-step SOC-feasible power range."""
        P_actual_min = -self.SOC * self.E_max / self.dt
        P_actual_max = (1.0 - self.SOC) * self.E_max / self.dt
        P_actual_target = float(
            np.clip(P_actual, P_actual_min, P_actual_max)
        )
        P_actual_clip_amount = abs(P_actual - P_actual_target)

        if P_actual_clip_amount > 0.0:
            P = self._grid_power_for_actual_power(P_actual_target, P)
            P_actual, arg, efficiency_factor = self._actual_power_mapping(P)
        else:
            P_actual, arg, efficiency_factor = self._actual_power_mapping(P)

        return (
            float(P),
            float(P_actual),
            float(arg),
            float(efficiency_factor),
            float(P_actual_min),
            float(P_actual_max),
            float(P_actual_clip_amount),
        )

    def _terminal_soc_reachability(self):
        """Return the SOC floor required to keep the terminal target reachable."""
        remaining_steps = max(self.T - self.t, 0)
        max_charge_soc_per_step = self.eta * self.P_max * self.dt / self.E_max
        terminal_soc_floor = max(
            0.0,
            self.terminal_soc_min
            - remaining_steps * max_charge_soc_per_step,
        )
        reachability_violation = max(terminal_soc_floor - self.SOC, 0.0)
        return (
            float(terminal_soc_floor),
            float(reachability_violation),
            float(max_charge_soc_per_step),
        )

    def _forecast_window(self):
        """Return the complete configured forecast at every environment step."""
        forecast = self.prices[: self.forecast_h]
        return np.tanh(forecast / self.price_scale).astype(np.float32)

    def _obs(self):
        idx = min(self.t, self.T - 1)
        p = self.prices[idx]
        norm_price = float(np.tanh(p / self.price_scale))
        denom = max(self.T - 1, 1)
        norm_time = float(min(self.t, self.T - 1) / denom)
        forecast = self._forecast_window()
        return np.concatenate(
            (np.array([self.SOC, norm_price, norm_time], dtype=np.float32), forecast)
        )

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
        P_requested = P
        P_actual_requested, _, _ = self._actual_power_mapping(P_requested)
        (
            P,
            P_actual,
            arg,
            efficiency_factor,
            P_actual_min,
            P_actual_max,
            P_actual_clip_amount,
        ) = self._limit_power_to_available_energy(P_requested, P_actual_requested)
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
        # P_actual_violation = max(abs(P_actual) - self.P_max, 0.0)
        # reward -= 10 * P_actual_violation
        (
            terminal_soc_floor,
            terminal_soc_reachability_violation,
            max_charge_soc_per_step,
        ) = self._terminal_soc_reachability()
        terminal_soc_reachability_penalty = 0.0
        if not terminated:
            terminal_soc_reachability_penalty = (
                self.terminal_soc_reachability_penalty
                * terminal_soc_reachability_violation
            )
            reward -= terminal_soc_reachability_penalty
        terminal_soc_violation = 0.0
        terminal_penalty = 0.0
        if terminated:
            terminal_soc_violation = max(self.terminal_soc_min - self.SOC, 0.0)
            terminal_penalty = self.terminal_soc_penalty * terminal_soc_violation
            reward -= terminal_penalty
        truncated = False
        info = {
            "P": P,
            "P_actual": P_actual,
            "P_requested": P_requested,
            "P_actual_requested": P_actual_requested,
            "P_actual_min": P_actual_min,
            "P_actual_max": P_actual_max,
            "P_actual_clipped": P_actual_clip_amount > 0.0,
            "P_actual_clip_amount": P_actual_clip_amount,
            "arg": arg,
            "efficiency_factor": efficiency_factor,
            "u": a,
            "price": price,
            "SOC": self.SOC,
            "cost": cost,
            "acc_cost": self.acc_cost,
            "terminal_soc_floor": terminal_soc_floor,
            "max_charge_soc_per_step": max_charge_soc_per_step,
            "terminal_soc_reachability_violation": terminal_soc_reachability_violation,
            "terminal_soc_reachability_penalty": terminal_soc_reachability_penalty,
            "terminal_soc_min": self.terminal_soc_min,
            "terminal_soc_violation": terminal_soc_violation,
            "terminal_penalty": terminal_penalty,
            "grid_reward": grid_reward,
            "raw_reward": reward,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        print(
            f"t={self.t} SOC={self.SOC:.3f} P={self._last_P:.1f} P_actual={self._last_P_actual:.1f}"
        )


class RandomWindowEnv(gym.Env):
    """Wrapper that samples random windows from full price data on each reset"""

    def __init__(self, prices_full, window_len=24, price_scale=None, **kwargs):
        self.prices_full = prices_full
        self.window_len = window_len
        self.max_start = len(prices_full) - window_len
        self.kwargs = kwargs
        self.price_scale = price_scale
        # Create a dummy env to get spaces
        dummy = BatteryEnv(
            prices_full[:window_len], price_scale=self.price_scale, **kwargs
        )
        self.observation_space = dummy.observation_space
        self.action_space = dummy.action_space
        self.metadata = dummy.metadata
        self._sample_new_window()

    def _sample_new_window(self):
        start = np.random.randint(0, self.max_start + 1)
        window_prices = self.prices_full[start : start + self.window_len]
        self.env = BatteryEnv(
            window_prices, price_scale=self.price_scale, **self.kwargs
        )

    def reset(self, **kwargs):
        self._sample_new_window()
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
