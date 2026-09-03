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
         R_applied_prev_norm,
         R_projection_delta_prev_norm,
         f4_residual_prev_norm,
         time_index_norm,
         terminal_temperature_gap_norm,
         beta_command_prev,
         m_I_prev_norm,
         forecast(g_grid), forecast(P_WT)
     ]

     -------------------------------------------------------------------------------
     ACTION VECTOR
     -------------------------------------------------------------------------------
     In the mathematical model:
     a_k = [beta_command_k, R_k, m_I_k]
     beta_command_k is a signed storage-mode command in [-1, 1].  Its three
     operating bands are:

       [-1, -0.05): discharge; beta1 is 1 and beta2 ramps from 0 to 1
       [-0.05, 0.05]: idle; both beta1 and beta2 are 1
       (0.05, 1]   : charge; beta1 ramps from 1 to 0 and beta2 is 1

     The ramp magnitude increases with distance from the idle band, so -1 and
     +1 respectively request full discharge and full charge.  beta1 and beta2
     are physical bypass coefficients, not direct flow fractions: a value of 1
     closes its corresponding storage flow in F6/F7.
     R_k is the HTHP surrogate variable (compressor speed),
     treated as a command and projected to the nearest F4-feasible value when
     possible. The applied value and projection feedback enter the next observation.
     Here beta1_k and beta2_k are the bypass variables x1 and x2,
     T_I_k is derived by F5, and T_III_k is the cold-side inlet temperature
     (fixed).
    """

    metadata = {"render_modes": ["human"]}  # noqa: RUF012

    def __init__(
        self,
        g_grid,  # grid electricity price (Pe)
        P_WT,  # wind turbine power availability (WKA)
        forecast_h=24,  # forecast horizon in number of steps (Time)
        Delta_t=3600.0,  # time step duration in seconds (deltaT)
        # TES parameters
        T_s_0=250.0,  # initial storage temperature in Celsius (S0)
        T_s_min=183.0,  # minimum storage temperature in Celsius (S_min)
        T_s_max=324.0,  # maximum storage temperature in Celsius (S_max)
        m_s=6.0e5,  # storage mass in kg (Ms)
        c_p_s=1.025,  # specific heat capacity of storage (in kJ/kg-K) (cp)
        epsilon_ch=0.9,  # charging efficiency (effCh)
        epsilon_dch=0.9,  # discharging efficiency (effDch)
        # oil/working-fluid side
        T_III_k=75.0,  # cold-side inlet temperature (fixed)
        c_p_f=2.21,  # specific heat capacity of working fluid (in kJ/kg-K) (cf)
        P_HP_max=5000.0,  # HTHP electric power upper bound in Wattes
        # scaling
        price_scale=None,
        wind_scale=None,
        # reward penalties
        lambda_bound=20.0,  # penalty for storage temperature bound violations
        lambda_terminal=3000.0,  # penalty for ending away from cyclic terminal target
        lambda_coupling=500.0,  # penalty for F4 coupling residuals
        lambda_grid=3000.0,  # penalty for grid power bound violations
        lambda_T_I=10.0,  # penalty for T_I bound violations
        lambda_flow=5000.0,  # penalty for Qch/Qdch flow bound violations
        eps_comp=1.0e-6,  # small tolerance for the Qch*Qdch complementarity constraint
        Q_flow_max=5000.0,  # Qch/Qdch upper bound from the NLP model
        enforce_cyclic_boundary=True,  # terminal storage constraint / terminal_penalty_k: return to T_s_0
    ):
        super().__init__()

        self.g_grid = np.asarray(g_grid, dtype=np.float32)
        self.P_WT = np.asarray(P_WT, dtype=np.float32)
        self.K = len(self.g_grid)  # number of timesteps in the episode (horizon length)

        if len(self.P_WT) != len(self.g_grid):
            raise ValueError("g_grid and P_WT must have same length")

        self.forecast_h = int(forecast_h)
        if self.forecast_h <= 0:
            raise ValueError("forecast_h must be positive.")
        if self.forecast_h > self.K:
            raise ValueError(
                "forecast_h cannot be larger than the supplied price and wind trajectories."
            )
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

        self.lambda_bound = float(lambda_bound)
        self.lambda_terminal = float(lambda_terminal)
        self.lambda_coupling = float(lambda_coupling)
        self.lambda_T_I = float(lambda_T_I)
        self.lambda_flow = float(lambda_flow)
        self.lambda_grid = float(lambda_grid)
        self.eps_comp = float(eps_comp)
        self.Q_flow_max = float(Q_flow_max)
        self.enforce_cyclic_boundary = bool(enforce_cyclic_boundary)

        # Observation = state, prior control values, time/terminal context, and
        # the same complete forecast at every step.
        base_low = np.array(
            [0.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0],
            dtype=np.float32,
        )
        base_high = np.ones(11, dtype=np.float32)
        fc_low = -np.ones(2 * self.forecast_h, dtype=np.float32)
        fc_high = np.ones(2 * self.forecast_h, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, fc_low]),
            high=np.concatenate([base_high, fc_high]),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.8, 5.0], dtype=np.float32),
            high=np.array([1.0, 1.53, 16.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset(seed=42)

    @staticmethod
    def _norm_signed(x, scale):
        return float(np.tanh(x / max(scale, 1e-6)))

    @staticmethod
    def _norm_01(x, lo, hi):
        return float(np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    @staticmethod
    def _map_beta_command(beta_command):
        """Map a signed command to physically feasible bypass coefficients.

        For the original process equations, ``Q_ch`` is proportional to
        ``1 - beta1`` and ``Q_dch`` to ``1 - beta2``.  Therefore the inactive
        branch must have a beta value of 1.  This mapping enforces both F10
        (beta1 + beta2 >= 1) and F11 (no simultaneous charge/discharge). The
        returned mode is -1 for discharge, 0 for idle, and 1 for charge.
        """
        beta_command = float(np.clip(beta_command, -1.0, 1.0))
        idle_half_width = 0.05
        active_band_width = 1.0 - idle_half_width

        if beta_command < -idle_half_width:
            # -1 -> full discharge (beta2=0), -0.05 -> idle (beta2=1).
            discharge_fraction = (-beta_command - idle_half_width) / active_band_width
            beta2 = 1.0 - discharge_fraction
            return 1.0, float(np.clip(beta2, 0.0, 1.0)), -1
        if beta_command > idle_half_width:
            # 0.05 -> idle (beta1=1), 1 -> full charge (beta1=0).
            charge_fraction = (beta_command - idle_half_width) / active_band_width
            beta1 = 1.0 - charge_fraction
            return float(np.clip(beta1, 0.0, 1.0)), 1.0, 1

        return 1.0, 1.0, 0

    @staticmethod
    def _HTHP_T_II_coefficients(T_I_k, m_I_k, T_III_k):
        """Return ascending polynomial coefficients for T_II as a function of R."""
        T_I = float(T_I_k)
        m_I = float(m_I_k)
        T_III = float(T_III_k)
        return np.array(
            [
                95.9612
                + 0.93433 * T_I
                - 0.327753 * m_I
                + 0.0146542 * T_III
                + 0.00104853 * T_I**2
                + 0.0211819 * T_I * m_I
                + 1.04924 * m_I**2
                - 0.00388073 * m_I * T_III
                - 0.00148575 * T_I * m_I**2
                - 0.0405702 * m_I**3,
                -271.354
                - 0.706122 * T_I
                - 29.4801 * m_I
                + 0.0595068 * T_III
                - 0.000716825 * T_I**2
                + 0.0229386 * T_I * m_I
                + 0.881391 * m_I**2,
                562.428 + 0.203578 * T_I - 2.18172 * m_I,
                -151.476,
            ],
            dtype=float,
        )
        

    def _project_R_for_f4(
        self,
        R_commanded_k,
        beta1_k,
        T_0_k,
        T_2_k,
        T_I_k,
        m_I_k,
    ):
        """Project R to the nearest value satisfying F4, when one exists."""
        denominator = 1.0 - self.epsilon_ch + self.epsilon_ch * beta1_k
        T_II_required_k = (
            T_2_k - self.epsilon_ch * T_0_k * (1.0 - beta1_k)
        ) / max(denominator, 1e-6)
        coefficients = self._HTHP_T_II_coefficients(
            T_I_k,
            m_I_k,
            self.T_III_k,
        )
        root_coefficients = coefficients.copy()
        root_coefficients[0] -= T_II_required_k
        roots = np.roots(root_coefficients[::-1])
        feasible_roots = [
            float(np.clip(root.real, 0.8, 1.53))
            for root in roots
            if abs(root.imag) <= 1e-7 and 0.8 - 1e-9 <= root.real <= 1.53 + 1e-9
        ]

        if feasible_roots:
            R_applied_k = min(
                feasible_roots,
                key=lambda candidate: abs(candidate - R_commanded_k),
            )
        else:
            derivative_roots = np.roots(
                [3.0 * coefficients[3], 2.0 * coefficients[2], coefficients[1]]
            )
            candidates = [0.8, 1.53]
            candidates.extend(
                float(root.real)
                for root in derivative_roots
                if abs(root.imag) <= 1e-7 and 0.8 <= root.real <= 1.53
            )
            R_applied_k = min(
                candidates,
                key=lambda candidate: (
                    abs(np.polynomial.polynomial.polyval(candidate, coefficients) - T_II_required_k),
                    abs(candidate - R_commanded_k),
                ),
            )

        T_II_projected_k = float(
            np.polynomial.polynomial.polyval(R_applied_k, coefficients)
        )
        projection_feasible_k = bool(
            abs(T_II_projected_k - T_II_required_k) <= 1e-5
        )
        projection_applied_k = bool(
            not np.isclose(R_applied_k, R_commanded_k, atol=1e-6, rtol=0.0)
        )
        return (
            float(R_applied_k),
            projection_applied_k,
            projection_feasible_k,
        )

    def _forecast_window(self):
        """Return the complete configured price and wind forecast at every step."""
        g = self.g_grid[: self.forecast_h]
        pwt = self.P_WT[: self.forecast_h]

        out = np.empty(2 * self.forecast_h, dtype=np.float32)
        out[0::2] = np.tanh(g / self.price_scale)
        out[1::2] = np.tanh(pwt / self.wind_scale)
        return out

    def _obs(self):
        k = min(self.k, self.K - 1)

        temperature_range = max(self.T_s_max - self.T_s_min, 1e-6)
        terminal_temperature_gap = float(
            np.clip((self.T_s_0 - self.T_s_k) / temperature_range, -1.0, 1.0)
        )
        time_index = float(self.k / max(self.K - 1, 1))

        base = np.array(
            [
                self._norm_01(self.T_s_k, self.T_s_min, self.T_s_max),
                self._norm_signed(self.g_grid[k], self.price_scale),
                self._norm_01(self.P_WT[k], 0.0, max(self.wind_scale, 1e-6)),
                self._last_mode,
                self._norm_01(self._last_R_k, 0.8, 1.53),
                self._last_R_projection_delta_k / (1.53 - 0.8),
                float(
                    np.clip(
                        self._last_f4_residual_k / temperature_range,
                        -1.0,
                        1.0,
                    )
                ),
                time_index,
                terminal_temperature_gap,
                self._last_beta_command_k,
                self._norm_01(self._last_m_I_k, 5.0, 16.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, self._forecast_window()])

    @classmethod
    def _HTHP_surrogate(cls, T_I_k, m_I_k, T_III_k, R_k):
        """Return electric power and hot-side outlet temperature (F1 and F3)."""
        T_I = float(T_I_k)
        m_I = float(m_I_k)
        T_III = float(T_III_k)
        R = float(np.clip(R_k, 0.8, 1.53))

        # F1: model.T1Out[t] == ...
        T_II_k = float(
            np.polynomial.polynomial.polyval(
                R,
                cls._HTHP_T_II_coefficients(T_I, m_I, T_III),
            )
        )

        # F3: model.P[t] + model.WKA[t] == ...
        P_HP = 3 * (
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
        return float(P_HP), T_II_k

    @staticmethod
    def _SG_surrogate(m_I_k):
        """Steam generator surrogate for F12, F13."""
        m_I = float(m_I_k)

        # F12: model.T4[t] == -188.403/(model.M1[t]*3) + 196.3
        T_3_k = -188.403 / (m_I * 3) + 196.3

        # F13: model.T3[t] == 201.915 + 1819.32/(model.M1[t]*3)
        T_2_k = 201.915 + 1819.32 / (m_I * 3)

        return T_2_k, T_3_k

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        self.T_s_k = self.T_s_0
        self._last_mode = 0
        self._last_R_k = 0.5 * (0.8 + 1.53)
        self._last_R_projection_delta_k = 0.0
        self._last_f4_residual_k = 0.0
        self._last_beta_command_k = 0.0
        self._last_m_I_k = 0.5 * (5.0 + 16.0)
        self._last_info = {}
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )

        beta_command_k = float(np.clip(action[0], -1.0, 1.0))
        beta1_k, beta2_k, mode_k = self._map_beta_command(beta_command_k)
        R_commanded_k = float(action[1])
        m_I_k = float(action[2])
        # C 15 (bounds)
        R_commanded_k = float(np.clip(R_commanded_k, 0.8, 1.53))
        m_I_k = float(np.clip(m_I_k, 5.0, 16.0))

        k = self.k
        g_grid_k = float(self.g_grid[k])
        P_WT_k = float(self.P_WT[k])

        # Initialize states/flows
        # F14: T_s_0 = T_tilde_0 and cyclic terminal condition is handled via terminal penalty
        T_0_k = self.T_s_k  # previous storage temperature

        # F12-F13: Steam generator surrogate. This surrogate depends on m_I_k only.
        T_2_k, T_3_k = self._SG_surrogate(m_I_k)

        # NLP variable mapping from cookie_model/mapping.py:
        # model.T2  -> T_1_k  (F8)
        # model.T3  -> T_2_k  (F13 / F4 residual check)
        # model.T4  -> T_3_k  (F12)
        # model.T5  -> T_4_k  (F9)
        # F9: T_4_k = T_3_k - epsilon_dch * (T_3_k - T_s_(k-1))
        T_4_k = T_3_k - self.epsilon_dch * (T_3_k - T_0_k)

        # F5
        T_I_k = float(T_3_k * beta2_k + T_4_k * (1.0 - beta2_k))

        T_I_bound_violation_k = max(177.0 - T_I_k, 0.0) + max(T_I_k - 250.0, 0.0)

        (
            R_applied_k,
            f4_projection_applied_k,
            f4_projection_feasible_k,
        ) = self._project_R_for_f4(
            R_commanded_k=R_commanded_k,
            beta1_k=beta1_k,
            T_0_k=T_0_k,
            T_2_k=T_2_k,
            T_I_k=T_I_k,
            m_I_k=m_I_k,
        )
        R_projection_delta_k = R_applied_k - R_commanded_k

        # F1/F3: HTHP surrogate, evaluated with the F5-consistent T_I_k.
        P_HP_k, T_II_k = self._HTHP_surrogate(
            T_I_k=T_I_k,
            m_I_k=m_I_k,
            T_III_k=self.T_III_k,
            R_k=R_applied_k,
        )
        # F8: T_1_k = T_II_k - epsilon_ch * (T_II_k - T_s_(k-1))
        T_1_k = T_II_k - self.epsilon_ch * (T_II_k - T_0_k)

        q_ch_base_k = (T_II_k - T_1_k) * m_I_k * 3 * self.c_p_f
        q_dch_base_k = (T_4_k - T_3_k) * m_I_k * 3 * self.c_p_f

        # F6: Q_s_ch_k = 3 * m_II_k * c_p_f * (T_II_k - T_1_k) * (1 - beta1_k)
        # F7: Q_s_dch_k = 3 * m_I_k * c_p_f * (T_4_k - T_3_k) * (1 - beta2_k)
        Q_s_ch_k = float(q_ch_base_k * (1.0 - beta1_k))
        Q_s_dch_k = float(q_dch_base_k * (1.0 - beta2_k))

        storage_factor_k = self.Delta_t / max(self.m_s * self.c_p_s, 1e-6)

        # F4: T_2_k = T_II_k * beta1_k + T_1_k * (1 - beta1_k)
        # F5: T_I_k = T_3_k * beta2_k + T_4_k * (1 - beta2_k)
        f4_residual_k = T_2_k - (T_1_k * (1.0 - beta1_k) + T_II_k * beta1_k)

        # F10: x1 + x2 >= 1.  The signed-command map enforces this exactly.
        bypass_sum_violation_k = max(1.0 - (beta1_k + beta2_k), 0.0)

        # F11: Qch * Qdch <= EPS.  The inactive bypass is closed (beta=1),
        # so only one storage flow can be non-zero.
        complementarity_violation_k = max(
            Q_s_ch_k * Q_s_dch_k - self.eps_comp,
            0.0,
        )
        flow_bound_violation_k = (
            max(-Q_s_ch_k, 0.0)
            + max(-Q_s_dch_k, 0.0)
            + max(Q_s_ch_k - self.Q_flow_max, 0.0)
            + max(Q_s_dch_k - self.Q_flow_max, 0.0)
        )

        # mode_k uses the same signed convention as beta_command:
        # -1=discharge, 0=idle, and 1=charge.

        # F3: model.P[t] + model.WKA[t] == 3*(...)  ->  P_grid_k = P_HP_k - P_WT_k
        P_grid_k = P_HP_k - P_WT_k

        # Grid power bound: 0 <= P <= 5000
        grid_bound_violation_k = max(-P_grid_k, 0.0) + max(
            P_grid_k - self.P_HP_max, 0.0
        )

        # Storage update (explicit Euler step)
        T_s_next = T_0_k + (Q_s_ch_k - Q_s_dch_k) * storage_factor_k

        # F14: T_s_n = T_s_0 (soft cyclic penalty)
        state_violation_k = max(self.T_s_min - T_s_next, 0.0) + max(
            T_s_next - self.T_s_max, 0.0
        )
        self.T_s_k = float(T_s_next)

        # Objective reward
        cost_grid_k = (g_grid_k / 1000.0) * P_grid_k
        penalty_bound_k = self.lambda_bound * (
            state_violation_k / max(self.T_s_max - self.T_s_min, 1e-6)
        )
        penalty_T_I_k = self.lambda_T_I * T_I_bound_violation_k / (250.0 - 177.0)
        penalty_flow_k = self.lambda_flow * flow_bound_violation_k / self.Q_flow_max
        penalty_coupling_k = self.lambda_coupling * (
            abs(f4_residual_k) / (self.T_s_max - self.T_s_min)
        )
        penalty_grid_k = self.lambda_grid * (
            grid_bound_violation_k / max(self.P_HP_max, 1e-6)
        )
        reward_k = -cost_grid_k
        reward_k -= penalty_bound_k
        reward_k -= penalty_T_I_k
        reward_k -= penalty_flow_k
        reward_k -= penalty_coupling_k
        reward_k -= penalty_grid_k

        self.k += 1
        terminated = self.k >= self.K
        truncated = False

        # Constraint-14 cyclic boundary handling
        if terminated and self.enforce_cyclic_boundary:
            terminal_penalty_k = (
                self.lambda_terminal
                * abs(self.T_s_k - self.T_s_0)
                / max(self.T_s_max - self.T_s_min, 1e-6)
            )
            reward_k -= terminal_penalty_k
        else:
            terminal_penalty_k = 0.0

        self._last_mode = mode_k
        self._last_R_k = R_applied_k
        self._last_R_projection_delta_k = R_projection_delta_k
        self._last_f4_residual_k = f4_residual_k
        self._last_beta_command_k = beta_command_k
        self._last_m_I_k = m_I_k
        info = {
            "beta_command_k": beta_command_k,
            "beta1_k": beta1_k,
            "beta2_k": beta2_k,
            "R_k": R_applied_k,
            "R_commanded_k": R_commanded_k,
            "R_applied_k": R_applied_k,
            "R_projection_delta_k": R_projection_delta_k,
            "f4_projection_applied_k": f4_projection_applied_k,
            "f4_projection_feasible_k": f4_projection_feasible_k,
            "m_I_k": m_I_k,
            "T_I_k": T_I_k,
            "T_s_k": self.T_s_k,
            "Q_s_ch_k": Q_s_ch_k,
            "Q_s_dch_k": Q_s_dch_k,
            "f4_residual_k": f4_residual_k,
            "bypass_sum_violation_k": bypass_sum_violation_k,
            "complementarity_violation_k": complementarity_violation_k,
            "flow_bound_violation_k": flow_bound_violation_k,
            "T_I_bound_violation_k": T_I_bound_violation_k,
            "P_HP_k": P_HP_k,
            "P_grid_k": P_grid_k,
            "P_grid_max": self.P_HP_max,
            "P_WT_k": P_WT_k,
            "grid_bound_violation_k": grid_bound_violation_k,
            "mode_k": mode_k,
            "state_violation_k": state_violation_k,
            "cost_grid_k": cost_grid_k,
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
