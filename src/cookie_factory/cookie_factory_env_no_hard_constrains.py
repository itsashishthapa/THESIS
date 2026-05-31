import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SteamEnv(gym.Env):
    """Cookie factory environment with reward-based constraint handling.

    Operational constraints are not repaired. A submitted action outside the
    declared action space is clipped before it reaches the surrogate models and
    receives an action-bound penalty.

    Equation-to-code mapping:
        Objective: minimize sum_k P_grid_k * g_grid_k * Delta_t.
        P_grid_k + P_WT_k = 3 * F_HTHP(T_I_k, m_I_k, T_III_k, R_k)
        T_II_k = F_HTHX(T_I_k, m_I_k, T_III_k, R_k)
        T_2_k = T_II_k * beta1_k + T_1_k * (1 - beta1_k)
        T_I_k = T_3_k * beta2_k + T_4_k * (1 - beta2_k)
        Q_s_ch_k = 3 * m_I_k * c_p_f * (T_II_k - T_1_k) * (1 - beta1_k)
        Q_s_dch_k = 3 * m_I_k * c_p_f * (T_4_k - T_3_k) * (1 - beta2_k)
        T_s_k = T_s_(k-1) + (Q_s_ch_k - Q_s_dch_k) * Delta_t / (m_s * c_p_s)

    Constraints represented as reward penalties:
        beta1_k + beta2_k >= 1
        max(Q_s_ch_k, 0) * max(Q_s_dch_k, 0) <= eps_comp
        Q_s_ch_k, Q_s_dch_k in [0, Q_flow_max]
        T_s_k in [T_s_min, T_s_max]
        P_grid_k in [0, P_HP_max]
        F4/F5 coupling residuals and terminal T_s_n = T_s_0

    Action:
        [beta1_k, beta2_k, R_k, m_I_k, T_I_k]

    Observation:
        [T_s_k_norm, g_grid_k_norm, P_WT_k_norm, mode_hint_prev,
         forecast(g_grid), forecast(P_WT)]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        g_grid,  # grid electricity price trajectory
        P_WT,  # available wind turbine power trajectory
        forecast_h=24,  # forecast horizon in steps
        Delta_t=3600.0,  # time-step duration in seconds
        # Thermal energy storage parameters
        T_s_0=250.0,  # initial storage temperature in Celsius
        T_s_min=183.0,  # nominal minimum storage temperature
        T_s_max=324.0,  # nominal maximum storage temperature
        m_s=6.0e5,  # storage mass in kg
        c_p_s=1.025,  # storage specific heat capacity in kJ/(kg K)
        epsilon_ch=0.9,  # charging efficiency
        epsilon_dch=0.9,  # discharging efficiency
        # Process side and HTHP surrogate parameters
        T_III_k=75.0,  # fixed cold-side inlet temperature
        c_p_f=2.21,  # working-fluid specific heat capacity in kJ/(kg K)
        P_HP_max=5000.0,  # maximum allowed grid/HTHP power
        # Observation scaling
        price_scale=None,
        wind_scale=None,
        # Soft constraint penalty weights
        lambda_mode=5.0,
        lambda_bound=20.0,
        lambda_grid=10.0,
        lambda_action_bound=10.0,
        lambda_comp=50000.0,
        lambda_coupling=500.0,
        lambda_terminal=50000.0,
        lambda_flow=20000.0,
        # Constraint limits
        eps_comp=1.0e-6,
        Q_flow_max=5000.0,
        enforce_cyclic_boundary=True,  # penalize terminal departure from T_s_0
    ):
        super().__init__()

        self.g_grid = np.asarray(g_grid, dtype=np.float32)
        self.P_WT = np.asarray(P_WT, dtype=np.float32)
        self.K = len(self.g_grid)
        if self.K == 0:
            raise ValueError("g_grid and P_WT must not be empty")
        if len(self.P_WT) != self.K:
            raise ValueError("g_grid and P_WT must have same length")

        self.forecast_h = int(forecast_h)
        self.Delta_t = float(Delta_t)
        self.T_s_0 = float(T_s_0)
        self.T_s_min = float(T_s_min)
        self.T_s_max = float(T_s_max)
        self.m_s = float(m_s)
        self.c_p_s = float(c_p_s)
        self.epsilon_ch = float(epsilon_ch)
        self.epsilon_dch = float(epsilon_dch)
        self.T_III_k = float(T_III_k)
        self.c_p_f = float(c_p_f)
        self.P_HP_max = float(P_HP_max)
        self.price_scale = (
            float(price_scale)
            if price_scale is not None
            else float(np.percentile(np.abs(self.g_grid), 95) + 1e-6)
        )
        self.wind_scale = (
            float(wind_scale)
            if wind_scale is not None
            else float(np.percentile(np.abs(self.P_WT), 95) + 1e-6)
        )
        self.lambda_mode = float(lambda_mode)
        self.lambda_bound = float(lambda_bound)
        self.lambda_terminal = float(lambda_terminal)
        self.lambda_coupling = float(lambda_coupling)
        self.lambda_comp = float(lambda_comp)
        self.lambda_grid = float(lambda_grid)
        self.lambda_flow = float(lambda_flow)
        self.lambda_action_bound = float(lambda_action_bound)
        self.eps_comp = float(eps_comp)
        self.Q_flow_max = float(Q_flow_max)
        self.enforce_cyclic_boundary = bool(enforce_cyclic_boundary)

        # Storage is unbounded in observations because bounds are soft.
        base_low = np.array([-np.inf, -1.0, 0.0, -1.0], dtype=np.float32)
        base_high = np.array([np.inf, 1.0, 1.0, 1.0], dtype=np.float32)
        forecast_low = -np.ones(2 * self.forecast_h, dtype=np.float32)
        forecast_high = np.ones(2 * self.forecast_h, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, forecast_low]),
            high=np.concatenate([base_high, forecast_high]),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.8, 5.0, 177.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.53, 16.0, 250.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset(seed=42)

    @staticmethod
    def _norm_signed(x, scale):
        return float(np.tanh(x / max(scale, 1e-6)))

    @staticmethod
    def _norm_01(x, lo, hi, clip=True):
        value = (x - lo) / max(hi - lo, 1e-6)
        return float(np.clip(value, 0.0, 1.0) if clip else value)

    def _forecast_window(self):
        k = min(self.k, self.K - 1)
        prices = self.g_grid[k + 1 : k + 1 + self.forecast_h]
        wind = self.P_WT[k + 1 : k + 1 + self.forecast_h]

        def pad(values, fallback):
            if len(values) < self.forecast_h:
                values = np.concatenate(
                    [
                        values,
                        np.full(
                            self.forecast_h - len(values),
                            fallback,
                            dtype=np.float32,
                        ),
                    ]
                )
            return values

        prices = pad(prices, self.g_grid[-1])
        wind = pad(wind, self.P_WT[-1])
        forecast = np.empty(2 * self.forecast_h, dtype=np.float32)
        forecast[0::2] = np.tanh(prices / self.price_scale)
        forecast[1::2] = np.tanh(wind / self.wind_scale)
        return forecast

    def _obs(self):
        k = min(self.k, self.K - 1)
        mode_hint = 0.0
        if self._last_mode == 1:
            mode_hint = -1.0
        elif self._last_mode == 2:
            mode_hint = 1.0

        base = np.array(
            [
                self._norm_01(self.T_s_k, self.T_s_min, self.T_s_max, clip=False),
                self._norm_signed(self.g_grid[k], self.price_scale),
                self._norm_01(self.P_WT[k], 0.0, max(self.wind_scale, 1e-6)),
                mode_hint,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, self._forecast_window()])

    def _hthp_surrogate(self, T_I_k, m_I_k, R_k):
        """Return HTHP electric power and hot-side outlet temperature."""
        T_I = float(T_I_k)
        m_I = float(m_I_k)
        T_III = self.T_III_k
        R = float(R_k)

        T_II_k = (
            95.9612
            + 0.93433 * T_I
            - 0.327753 * m_I
            + 0.0146542 * T_III
            - 271.354 * R
            + 0.00104853 * T_I**2
            + 0.0211819 * T_I * m_I
            - 0.706122 * T_I * R
            + 1.04924 * m_I**2
            - 0.00388073 * m_I * T_III
            - 29.4801 * m_I * R
            + 0.0595068 * T_III * R
            + 562.428 * R**2
            - 0.000716825 * T_I**2 * R
            - 0.00148575 * T_I * m_I**2
            + 0.0229386 * T_I * m_I * R
            + 0.203578 * T_I * R**2
            - 0.0405702 * m_I**3
            + 0.881391 * m_I**2 * R
            - 2.18172 * m_I * R**2
            - 151.476 * R**3
        )
        P_HP_k = 3 * (
            127.87
            + 2.06342 * T_I
            + 2.55723 * m_I
            + 0.756419 * T_III
            - 1164.84 * R
            - 0.0168942 * T_I * m_I
            - 2.60579 * T_I * R
            - 0.540713 * m_I**2
            + 13.3204 * m_I * R
            - 1.3829 * T_III * R
            + 1556.66 * R**2
        )
        return float(P_HP_k), float(T_II_k)

    @staticmethod
    def _steam_generator_temperatures(m_I_k):
        m_I = float(m_I_k)
        T_2_k = 201.915 + 1819.32 / (m_I * 3)
        T_3_k = -188.403 / (m_I * 3) + 196.3
        return float(T_2_k), float(T_3_k)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        self.T_s_k = self.T_s_0
        self._last_mode = 0
        self._last_P_grid = 0.0
        return self._obs(), {}

    def step(self, action):
        # The action box is the surrogate's valid operating domain.
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        action_span = self.action_space.high - self.action_space.low
        action_bound_violation = float(np.sum(np.abs(action - clipped_action) / action_span))
        beta1_k, beta2_k, R_k, m_I_k, T_I_k = (float(value) for value in clipped_action)

        k = self.k
        T_0_k = self.T_s_k
        # Steam generator outputs and TES interface temperatures (F8/F9/F12/F13).
        T_2_k, T_3_k = self._steam_generator_temperatures(m_I_k)
        T_4_k = T_3_k - self.epsilon_dch * (T_3_k - T_0_k)
        P_HP_k, T_II_k = self._hthp_surrogate(T_I_k, m_I_k, R_k)
        T_1_k = T_II_k - self.epsilon_ch * (T_II_k - T_0_k)

        # Storage charge/discharge flows (F6/F7).
        Q_s_ch_k = (T_II_k - T_1_k) * m_I_k * 3 * self.c_p_f * (1.0 - beta1_k)
        Q_s_dch_k = (T_4_k - T_3_k) * m_I_k * 3 * self.c_p_f * (1.0 - beta2_k)

        f4_residual = T_2_k - (T_1_k * (1.0 - beta1_k) + T_II_k * beta1_k)
        f5_residual = T_I_k - (T_3_k * beta2_k + T_4_k * (1.0 - beta2_k))
        bypass_violation = max(1.0 - beta1_k - beta2_k, 0.0)
        positive_Q_ch = max(Q_s_ch_k, 0.0)
        positive_Q_dch = max(Q_s_dch_k, 0.0)
        complementarity_violation = max(
            positive_Q_ch * positive_Q_dch - self.eps_comp,
            0.0,
        )
        simultaneous_flow = min(positive_Q_ch, positive_Q_dch)
        flow_violation = (
            max(-Q_s_ch_k, 0.0)
            + max(-Q_s_dch_k, 0.0)
            + max(Q_s_ch_k - self.Q_flow_max, 0.0)
            + max(Q_s_dch_k - self.Q_flow_max, 0.0)
        )

        # Grid balance (F3) and explicit storage update (F14 state transition).
        P_grid_k = P_HP_k - float(self.P_WT[k])
        grid_violation = max(-P_grid_k, 0.0) + max(P_grid_k - self.P_HP_max, 0.0)
        storage_factor = self.Delta_t / max(self.m_s * self.c_p_s, 1e-6)
        self.T_s_k = float(T_0_k + (Q_s_ch_k - Q_s_dch_k) * storage_factor)
        storage_violation = max(self.T_s_min - self.T_s_k, 0.0) + max(
            self.T_s_k - self.T_s_max, 0.0
        )

        # Objective plus soft penalties for every operational constraint.
        reward = -(float(self.g_grid[k]) / 1000.0) * P_grid_k
        reward -= self.lambda_mode * bypass_violation
        reward -= self.lambda_bound * storage_violation / max(
            self.T_s_max - self.T_s_min, 1e-6
        )
        reward -= self.lambda_coupling * (abs(f4_residual) + abs(f5_residual))
        # Use simultaneous flow for the reward; the raw product is too large and
        # makes SAC trade constraints erratically.
        reward -= self.lambda_comp * simultaneous_flow / max(self.Q_flow_max, 1e-6)
        reward -= self.lambda_grid * grid_violation / max(self.P_HP_max, 1e-6)
        reward -= self.lambda_flow * flow_violation / max(self.Q_flow_max, 1e-6)
        reward -= self.lambda_action_bound * action_bound_violation

        if Q_s_ch_k > 1e-9 and Q_s_dch_k <= 1e-9:
            self._last_mode = 1
        elif Q_s_dch_k > 1e-9 and Q_s_ch_k <= 1e-9:
            self._last_mode = 2
        elif abs(Q_s_ch_k) <= 1e-9 and abs(Q_s_dch_k) <= 1e-9:
            self._last_mode = 0
        else:
            self._last_mode = -1
        self._last_P_grid = P_grid_k

        self.k += 1
        terminated = self.k >= self.K
        if terminated and self.enforce_cyclic_boundary:
            reward -= self.lambda_terminal * abs(self.T_s_k - self.T_s_0) / max(
                self.T_s_max - self.T_s_min, 1e-6
            )

        return self._obs(), float(reward), terminated, False, {}

    def render(self):
        print(
            f"k={self.k} | T_s_k={self.T_s_k:.2f} C | mode={self._last_mode} | "
            f"P_grid_k={self._last_P_grid:.1f} W"
        )


class RandomWindowEnv(gym.Env):
    """Apply SteamEnv to random fixed-length slices of the data."""

    def __init__(self, g_grid_full, P_WT_full, window_len=24, **kwargs):
        self.g_grid_full = np.asarray(g_grid_full, dtype=np.float32)
        self.P_WT_full = np.asarray(P_WT_full, dtype=np.float32)
        if len(self.P_WT_full) != len(self.g_grid_full):
            raise ValueError("All full-series inputs must have the same length")
        if window_len > len(self.g_grid_full):
            raise ValueError("window_len cannot be larger than the available history")

        self.window_len = int(window_len)
        self.max_start = len(self.g_grid_full) - self.window_len
        self.kwargs = kwargs
        probe = SteamEnv(
            self.g_grid_full[: self.window_len],
            self.P_WT_full[: self.window_len],
            **kwargs,
        )
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space
        self.metadata = probe.metadata
        self._sample_new_window()

    def _sample_new_window(self):
        start = np.random.randint(0, self.max_start + 1)
        end = start + self.window_len
        self.env = SteamEnv(
            self.g_grid_full[start:end],
            self.P_WT_full[start:end],
            **self.kwargs,
        )

    def reset(self, **kwargs):
        self._sample_new_window()
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
