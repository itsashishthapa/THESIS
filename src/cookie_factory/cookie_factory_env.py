import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SteamEnv(gym.Env):
    """
    RL environment for the renewable steam generation problem

    -------------------------------------------------------------------------------
    EQUATION-TO-CODE MAPPING
    -------------------------------------------------------------------------------
    Obj: min J(P_grid) = sum_k P_grid_k * g_grid_k * Delta_t

    1) P_grid_k + P_WT_k = 3 * F_HTHP(T_I_k, m_I_k, T_III_k, R_k)
    2) T_II_k = F_HTHX(T_I_k, m_I_k, T_III_k, R_k)
    3) T_IV_k = F_LTHX(T_I_k, m_I_k, T_III_k, R_k)
    4) T_2_k = T_II_k * beta1_k + T_1_k * (1 - beta1_k)
    5) T_I_k = T_3_k * beta2_k + T_4_k * (1 - beta2_k)
    6) T_2_k = 201.92 + 1819.32 / (3 * m_II_k)
    7) T_3_k = 196.3 - 188.4 / (3 * m_I_k)
    8) Q_s_ch_k = 3 * m_II_k * c_p_f * (T_II_k - T_1_k) * (1 - beta1_k)
    9) Q_s_dch_k = 3 * m_I_k * c_p_f * (T_4_k - T_3_k) * (1 - beta2_k)
   10) Q_s_ch_k * Q_s_dch_k <= gamma
   11) T_1_k = T_II_k - epsilon_ch * (T_II_k - T_s_(k-1))
   12) T_4_k = T_3_k - epsilon_dch * (T_3_k - T_s_(k-1))
   13) T_s_k = T_s_(k-1) + (Q_s_ch_k - Q_s_dch_k) / (m_s * c_p_s) * Delta_t
   14) T_s_0 = T_tilde_0,  T_s_n = T_s_0
   15) T_I_k in [177, 250],  m_I_k in [5, 16]
   16) T_III_k in [60, 100], R_k in [0.8, 1.53]
   17) t_k = Delta_t * k

    -------------------------------------------------------------------------------
    OBSERVATION VECTOR
    -------------------------------------------------------------------------------
    obs_k = [
        T_s_k_norm,
        g_grid_k_norm,
        P_WT_k_norm,
        mode_hint_prev,
        forecast(g_grid), forecast(P_WT)
    ]

    -------------------------------------------------------------------------------
    ACTION VECTOR
    -------------------------------------------------------------------------------
    In the mathematical model:
    a_k = [beta1_k, beta2_k, R_k, m_I_k, T_I_k]
    R_k is HTHP surrogate variable (compressor speed) 
    where beta1_k and beta2_k are the bypass variables x1 and x2,
    T_I_k is the hot-side inlet temperature decision variable,
    and T_III_k is the cold-side inlet temperature (fixed).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        g_grid, # grid electricity price (Pe)
        P_WT, # wind turbine power availability (WKA)
        forecast_h=24, # forecast horizon in number of steps (Time)
        Delta_t=3600.0, # time step duration in seconds (deltaT)
        # TES parameters
        T_s_0=250.0, # initial storage temperature in Celsius (S0)
        T_s_min=183.0, # minimum storage temperature in Celsius (S_min)
        T_s_max=324.0, # maximum storage temperature in Celsius (S_max)
        m_s=6.0e5, # storage mass in kg (Ms)
        c_p_s=1.025, # specific heat capacity of storage (in kJ/kg-K) (cp)
        epsilon_ch=0.9, # charging efficiency (effCh)
        epsilon_dch=0.9, # discharging efficiency (effDch)
        # oil/working-fluid side
        T_III_k=75.0, # cold-side inlet temperature (fixed)
        c_p_f=2.21, # specific heat capacity of working fluid (in kJ/kg-K) (cf)
        P_HP_max=5000.0, # HTHP electric power upper bound in Wattes
        # scaling
        price_scale=None,
        wind_scale=None,
        # reward penalties
        lambda_mode=5.0, # penalty for invalid bypass/mode behavior
        lambda_bound=20.0, # penalty for storage temperature bound violations
        lambda_terminal=2.0, # penalty for ending away from cyclic terminal target
        lambda_coupling=2.0, # penalty for F4/F5 coupling residuals
        lambda_comp=2.0, # penalty for simultaneous charging and discharging
        lambda_grid=10.0, # penalty for grid power bound violations
        eps_comp=1.0e-6, # small tolerance for the Qch*Qdch complementarity constraint
        enforce_cyclic_boundary=True, # enforce final storage temperature to return to the initial value
    ):
        super().__init__()

        self.g_grid = np.asarray(g_grid, dtype=np.float32)
        self.P_WT = np.asarray(P_WT, dtype=np.float32)
        self.K = len(self.g_grid) # number of timesteps in the episode (horizon length)

        if len(self.P_WT) !=  len(self.g_grid):
            raise ValueError("g_grid and P_WT must have same length")

        self.forecast_h = int(forecast_h)
        self.Delta_t = float(Delta_t)

        # TES parameters
        self.T_s_0 = float(T_s_0)
        self.T_s_min = float(T_s_min)
        self.T_s_max = float(T_s_max)
        self.m_s = float(m_s)
        self.c_p_s = float(c_p_s)
        self.epsilon_ch = float(epsilon_ch)
        self.epsilon_dch = float(epsilon_dch)

        # Oil/working-fluid side parameters
        self.c_p_f = float(c_p_f)
        self.P_HP_max = float(P_HP_max)
        self.T_III_k = float(T_III_k)  # HTHP cold-side inlet temperature (fixed)

        self.price_scale = float(price_scale) if price_scale is not None else float(np.percentile(np.abs(self.g_grid), 95) + 1e-6)
        self.wind_scale = float(wind_scale) if wind_scale is not None else float(np.percentile(np.abs(self.P_WT), 95) + 1e-6)

        self.lambda_mode = float(lambda_mode)
        self.lambda_bound = float(lambda_bound)
        self.lambda_terminal = float(lambda_terminal)
        self.lambda_coupling = float(lambda_coupling)
        self.lambda_comp = float(lambda_comp)
        self.lambda_grid = float(lambda_grid)
        self.eps_comp = float(eps_comp)
        self.enforce_cyclic_boundary = bool(enforce_cyclic_boundary)

        # Observation = [T_s_k, g_grid_k, P_WT_k, last_mode] + forecast
        base_low = np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)
        base_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        fc_low = -np.ones(2 * self.forecast_h, dtype=np.float32)
        fc_high = np.ones(2 * self.forecast_h, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, fc_low]),
            high=np.concatenate([base_high, fc_high]),
            dtype=np.float32,
        )

        # Action = [beta1_k, beta2_k, R_k, m_I_k, T_I_k]
        # x1,x2 in [0,1], N in [0.8,1.53], M1 in [5,16], T_I in [177,250]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.8, 5.0, 177.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.53, 16.0, 250.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset(seed=42)

    def _norm_signed(self, x, scale):
        return float(np.tanh(x / max(scale, 1e-6)))

    def _norm_01(self, x, lo, hi):
        return float(np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    def _forecast_window(self):
        k = min(self.k, self.K - 1)
        g = self.g_grid[k + 1: k + 1 + self.forecast_h] # forecast horizon of grid prices
        pwt = self.P_WT[k + 1: k + 1 + self.forecast_h] # forecast horizon of wind turbine availability

        def pad(arr, pad_val):
            if len(arr) < self.forecast_h:
                arr = np.concatenate([arr, np.full(self.forecast_h - len(arr), pad_val, dtype=np.float32)])
            return arr

        g = pad(g, self.g_grid[-1])
        pwt = pad(pwt, self.P_WT[-1])

        out = np.empty(2 * self.forecast_h, dtype=np.float32)
        out[0::2] = np.tanh(g / self.price_scale)
        out[1::2] = np.tanh(pwt / self.wind_scale)
        return out

    def _obs(self):
        k = min(self.k, self.K - 1)
        mode_hint = 0.0
        if self._last_mode == 1:
            mode_hint = -1.0
        elif self._last_mode == 2:
            mode_hint = 1.0

        base = np.array([
            self._norm_01(self.T_s_k, self.T_s_min, self.T_s_max),
            self._norm_signed(self.g_grid[k], self.price_scale),
            self._norm_01(self.P_WT[k], 0.0, max(self.wind_scale, 1e-6)),
            mode_hint,
        ], dtype=np.float32)
        return np.concatenate([base, self._forecast_window()])

    def _HTHP_surrogate(self, T_I_k, m_I_k, T_III_k, R_k):
        """HTHP surrogate for F1, F2, F3."""
        T_I = float(T_I_k)
        m_I = float(m_I_k)
        T_III = float(T_III_k)
        R = float(np.clip(R_k, 0.8, 1.53))
        
        # F1: model.T1Out[t] == ...
        T_II_k = (
            95.9612 + 0.93433*T_I - 0.327753*m_I + 0.0146542*T_III - 271.354*R
            + 0.00104853*T_I**2 + 0.0211819*T_I*m_I - 0.706122*T_I*R + 1.04924*m_I**2
            - 0.00388073*m_I*T_III - 29.4801*m_I*R + 0.0595068*T_III*R + 562.428*R**2
            - 0.000716825*T_I**2*R - 0.00148575*T_I*m_I**2 + 0.0229386*T_I*m_I*R
            + 0.203578*T_I*R**2 - 0.0405702*m_I**3 + 0.881391*m_I**2*R - 2.18172*m_I*R**2
            - 151.476*R**3
        )
        
        # F2: model.T2Out[t] == ...
        T_IV_k = (
            93.3958 - 0.00692483*T_I - 0.770173*m_I + 1.30277*T_III - 183.866*R
            + 0.00313225*T_I*m_I + 0.234082*T_I*R + 0.106964*m_I**2
            - 2.34999*m_I*R - 0.555879*T_III*R + 30.2955*R**2
        )
        
        # F3: model.P[t] + model.WKA[t] == ...
        P_HP = 3 * (
            127.87 + 2.06342*T_I + 2.55723*m_I + 0.756419*T_III - 1164.84*R
            - 0.0168942*T_I*m_I - 2.60579*T_I*R - 0.540713*m_I**2 + 13.3204*m_I*R
            - 1.3829*T_III*R + 1556.66*R**2
        )
        P_HP_k = float(P_HP)
        
        return P_HP_k, T_IV_k, T_II_k

    def _SG_surrogate(self, T_in_SG_k, m_I_k):
        """Steam generator surrogate for F12, F13."""
        m_I = float(m_I_k)
        
        # F12: model.T4[t] == -188.403/(model.M1[t]*3) + 196.3
        T_4_k = -188.403 / (m_I * 3) + 196.3
        
        # F13: model.T3[t] == 201.915 + 1819.32/(model.M1[t]*3)
        T_3_k = 201.915 + 1819.32 / (m_I * 3)
        
        return T_3_k, T_4_k

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        self.T_s_k = self.T_s_0
        self._last_mode = 0
        self._last_info = {}
        return self._obs(), {}

    def step(self, action):
        # a_k = [beta1_k, beta2_k, R_k, m_I_k, T_I_k]
        beta1_k = float(np.clip(action[0], 0.0, 1.0))
        beta2_k = float(np.clip(action[1], 0.0, 1.0))
        # C 15 (bounds)
        R_k = float(np.clip(action[2], 0.8, 1.53))  
        m_I_k = float(np.clip(action[3], 5.0, 16.0))
        T_I_k = float(np.clip(action[4], 177.0, 250.0))
        T_III_k = self.T_III_k  # fixed

        k = self.k
        g_grid_k = float(self.g_grid[k])
        P_WT_k = float(self.P_WT[k])

        # F1-F3: HTHP surrogate
        P_HP_k, _, T_II_k = self._HTHP_surrogate(
            T_I_k=T_I_k,
            m_I_k=m_I_k,
            T_III_k=T_III_k,
            R_k=R_k,
        )

        # F12-F13: Steam generator surrogate
        T_SG_in, T_SG_out = self._SG_surrogate(T_in_SG_k=T_II_k, m_I_k=m_I_k)

        # Initialize states/flows
        # F14: T_s_0 = T_tilde_0 and cyclic terminal condition is handled via terminal penalty
        T_0_k = self.T_s_k  # previous storage temperature
        # NLP variable mapping from cookie_model/mapping.py:
        # model.T2  -> T_1_k  (F8)
        # model.T3  -> T_2_k  (F13 / F4 residual check)
        # model.T4  -> T_3_k  (F12)
        # model.T5  -> T_4_k  (F9)
        # F8: T_1_k = T_II_k - epsilon_ch * (T_II_k - T_s_(k-1))
        T_1_k = T_II_k - self.epsilon_ch * (T_II_k - T_0_k)
        T_2_k = T_SG_in
        T_3_k = float(T_SG_out)
        # F9: T_4_k = T_3_k - epsilon_dch * (T_3_k - T_s_(k-1))
        T_4_k = T_3_k - self.epsilon_dch * (T_3_k - T_0_k)

        # F6: Q_s_ch_k = 3 * m_II_k * c_p_f * (T_II_k - T_1_k) * (1 - beta1_k)
        Q_s_ch_k = (T_II_k - T_1_k) * m_I_k * 3 * self.c_p_f * (1.0 - beta1_k)
        # F7: Q_s_dch_k = 3 * m_I_k * c_p_f * (T_4_k - T_3_k) * (1 - beta2_k)
        Q_s_dch_k = (T_4_k - T_3_k) * m_I_k * 3 * self.c_p_f * (1.0 - beta2_k)

        # F4: T_2_k = T_II_k * beta1_k + T_1_k * (1 - beta1_k)
        # F5: T_I_k = T_3_k * beta2_k + T_4_k * (1 - beta2_k)
        f4_residual_k = T_2_k - (T_1_k * (1.0 - beta1_k) + T_II_k * beta1_k)
        f5_residual_k = T_I_k - (T_3_k * beta2_k + T_4_k * (1.0 - beta2_k))
        # F10: x1 + x2 >= 1
        bypass_sum_violation_k = max(1.0 - (beta1_k + beta2_k), 0.0)
        charge_discharge_product_k = Q_s_ch_k * Q_s_dch_k
        # F11: Qch * Qdch <= EPS
        complementarity_violation_k = max(charge_discharge_product_k - self.eps_comp, 0.0)

        # Diagnostic mode only (for observation hint)
        if Q_s_ch_k > 1e-9 and Q_s_dch_k <= 1e-9:
            mode_k = 1
        elif Q_s_dch_k > 1e-9 and Q_s_ch_k <= 1e-9:
            mode_k = 2
        elif abs(Q_s_ch_k) <= 1e-9 and abs(Q_s_dch_k) <= 1e-9:
            mode_k = 0
        else:
            mode_k = -1
        mode_violation_k = bypass_sum_violation_k
        
        # F3: model.P[t] + model.WKA[t] == 3*(...)  ->  P_grid_k = P_HP_k - P_WT_k
        P_grid_k = P_HP_k - P_WT_k
        P_spill_k = 0.0

        # Grid power bound: 0 <= P <= 5000
        grid_lower_violation_k = max(-P_grid_k, 0.0)
        grid_upper_violation_k = max(P_grid_k - self.P_HP_max, 0.0)
        grid_bound_violation_k = grid_lower_violation_k + grid_upper_violation_k

        # Storage update (explicit Euler step)
        T_s_next = T_0_k + (Q_s_ch_k - Q_s_dch_k) * self.Delta_t / max(self.m_s * self.c_p_s, 1e-6)

        # F14: T_s_n = T_s_0 (soft cyclic penalty)
        lower_violation_k = max(self.T_s_min - T_s_next, 0.0)
        upper_violation_k = max(T_s_next - self.T_s_max, 0.0)
        state_violation_k = lower_violation_k + upper_violation_k
        self.T_s_k = float(np.clip(T_s_next, self.T_s_min, self.T_s_max))

        # Objective reward
        cost_grid_k = (g_grid_k / 1000.0) * P_grid_k
        reward_k = -cost_grid_k
        reward_k -= self.lambda_mode * mode_violation_k
        reward_k -= self.lambda_bound * (state_violation_k / max(self.T_s_max - self.T_s_min, 1e-6))
        reward_k -= self.lambda_coupling * (abs(f4_residual_k) + abs(f5_residual_k))
        reward_k -= self.lambda_comp * complementarity_violation_k
        reward_k -= self.lambda_grid * (grid_bound_violation_k / max(self.P_HP_max, 1e-6))

        self.k += 1
        terminated = self.k >= self.K
        truncated = False

        # Constraint-14 cyclic boundary handling
        if terminated and self.enforce_cyclic_boundary:
            terminal_penalty_k = self.lambda_terminal * abs(self.T_s_k - self.T_s_0) / max(self.T_s_max - self.T_s_min, 1e-6)
            reward_k -= terminal_penalty_k
        else:
            terminal_penalty_k = 0.0

        # C17
        t_k = self.Delta_t * k

        self._last_mode = mode_k
        info = {
            "beta1_k": beta1_k,
            "beta2_k": beta2_k,
            "R_k": R_k,
            "m_I_k": m_I_k,
            "T_I_k": T_I_k,
            "T_s_k": self.T_s_k,
            "T_1_k": T_1_k,
            "T_2_k": T_2_k,
            "T_3_k": T_3_k,
            "T_4_k": T_4_k,
            "T_0_k": T_0_k,
            "T_SG_out": float(T_SG_out),
            "T_II_k": T_II_k,
            "Q_s_ch_k": Q_s_ch_k,
            "Q_s_dch_k": Q_s_dch_k,
            "charge_discharge_product_k": charge_discharge_product_k,
            "f4_residual_k": f4_residual_k,
            "f5_residual_k": f5_residual_k,
            "bypass_sum_violation_k": bypass_sum_violation_k,
            "complementarity_violation_k": complementarity_violation_k,
            "P_HP_k": P_HP_k,
            "P_grid_k": P_grid_k,
            "P_WT_k": P_WT_k,
            "P_spill_k": P_spill_k,
            "grid_lower_violation_k": grid_lower_violation_k,
            "grid_upper_violation_k": grid_upper_violation_k,
            "grid_bound_violation_k": grid_bound_violation_k,
            "g_grid_k": g_grid_k,
            "mode_k": mode_k,
            "mode_violation_k": mode_violation_k,
            "state_violation_k": state_violation_k,
            "cost_grid_k": cost_grid_k,
            "terminal_penalty_k": terminal_penalty_k,
            "t_k": t_k,
            "raw_reward": float(reward_k),
            # convenience aliases
            "beta1": beta1_k,
            "beta2": beta2_k,
            "T_s": self.T_s_k,
            "Q_s_ch": Q_s_ch_k,
            "Q_s_dch": Q_s_dch_k,
            "P_grid": P_grid_k,
        }
        self._last_info = info

        return self._obs(), float(reward_k), terminated, truncated, info

    def render(self):
        print(
            f"k={self.k} | T_s_k={self.T_s_k:.2f} C | mode={self._last_mode} | "
            f"P_grid_k={self._last_info.get('P_grid_k', 0.0):.1f} W"
        )


class RandomWindowEnv(gym.Env):
    """Random window wrapper"""

    def __init__(self, g_grid_full, P_WT_full, window_len=24, **kwargs):
        self.g_grid_full = np.asarray(g_grid_full, dtype=np.float32)
        self.P_WT_full = np.asarray(P_WT_full, dtype=np.float32)

        n = len(self.g_grid_full)
        if len(self.P_WT_full) != n:
            raise ValueError("All full-series inputs must have the same length")
        if window_len > n:
            raise ValueError("window_len cannot be larger than the available history")

        self.window_len = int(window_len)
        self.max_start = n - self.window_len
        self.kwargs = kwargs

        dummy = SteamEnv(
            self.g_grid_full[:window_len],
            self.P_WT_full[:window_len],
            **kwargs,
        )
        self.observation_space = dummy.observation_space
        self.action_space = dummy.action_space
        self.metadata = dummy.metadata
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
