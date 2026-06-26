"""
Coupled wind-speed and electricity-price model from Pilling et al. (2025).

The paper models the exogenous states as

    log W(t) = mu_W(t) + Y_W(t)
    S(t)     = mu_S(t) + Y_S(t)

where the residuals follow coupled Ornstein-Uhlenbeck dynamics. The price
residual is driven by its own Brownian shock and by the wind residual, producing
the wind-price dependence that was missing from the earlier electricity-only
model.

Time is measured in hours. Prices are in EUR/MWh.
"""

import math

import numpy as np


YEARLY_PERIOD_HOURS = 8760.0
DAILY_PERIOD_HOURS = 24.0
HALF_DAILY_PERIOD_HOURS = 12.0
DEFAULT_PRICE_MODEL_HOURS = 8784

# Seasonality constants for mu_S(t), Table G.1.
K0_S = -11.2038
K1_S = 4.2571
K2_S = -6.6642
K3_S = 30.4945
T1_S = -14782.5
T2_S = -6.7823
T3_S = -9.5016

# Seasonality constants for mu_W(t), Table G.1.
K0_W = 0.1357
K1_W = -0.328
K2_W = 1.6496
T1_W = 1034.1
T2_W = 1.1707

# Ornstein-Uhlenbeck constants, Table B.1.
LAMBDA_W = 0.1702
SIGMA_W = 0.2486
C_W = 0.5483
LAMBDA_S = 0.2534
SIGMA_S = 0.1072

# The paper's calibrated price residual is small on the plotted EUR/MWh scale.
# Keep this explicit scenario multiplier so generated price paths show visible
# stochastic variation while retaining the paper's coupled dynamics.
RESIDUAL_VOLATILITY_SCALE = 50.0
PRICE_VOLATILITY_SCALE = RESIDUAL_VOLATILITY_SCALE
WIND_VOLATILITY_SCALE = 1.0
PRICE_LEVEL_SHIFT = 30.0


def _return_scalar_if_scalar(value, reference):
    return float(value) if np.ndim(np.asarray(reference, dtype=float)) == 0 else value


def seasonality(hour):
    """Return the deterministic electricity-price seasonality mu_S(t)."""
    h = np.asarray(hour, dtype=float)
    two_pi = 2.0 * np.pi
    value = (
        K0_S
        + K1_S * np.cos(two_pi * (h - T1_S) / YEARLY_PERIOD_HOURS)
        + K2_S * np.cos(two_pi * (h - T2_S) / DAILY_PERIOD_HOURS)
        + K3_S * np.cos(two_pi * (h - T3_S) / HALF_DAILY_PERIOD_HOURS)
    )
    return _return_scalar_if_scalar(value, h)


def seasonal_mean(hour):
    """Return the mean electricity price used by the simulated process."""
    h = np.asarray(hour, dtype=float)
    value = seasonality(h) + PRICE_LEVEL_SHIFT
    return _return_scalar_if_scalar(value, h)


def wind_seasonality(hour):
    """Return the deterministic log-wind seasonality mu_W(t)."""
    h = np.asarray(hour, dtype=float)
    two_pi = 2.0 * np.pi
    value = (
        K0_W
        + K1_W * np.cos(two_pi * (h - T1_W) / YEARLY_PERIOD_HOURS)
        + K2_W * np.cos(two_pi * (h - T2_W) / DAILY_PERIOD_HOURS)
    )
    return _return_scalar_if_scalar(value, h)


def wind_seasonal_mean(hour):
    """Return the seasonal mean wind speed exp(mu_W(t))."""
    h = np.asarray(hour, dtype=float)
    value = np.exp(wind_seasonality(h))
    return _return_scalar_if_scalar(value, h)


def residual(price, hour):
    """Return the price residual Y_S(t) = S(t) - mu_S(t)."""
    return float(price - seasonal_mean(hour))


def wind_residual(wind_speed, hour):
    """Return the wind residual Y_W(t) = log(W(t)) - mu_W(t)."""
    wind_speed = max(float(wind_speed), 1e-9)
    return float(math.log(wind_speed) - wind_seasonality(hour))


def _effective_sigmas(wind_volatility_scale=1.0):
    return float(SIGMA_W) * float(wind_volatility_scale), float(SIGMA_S)


def ou_covariance(
    dt_hours=1.0,
    wind_volatility_scale=WIND_VOLATILITY_SCALE,
    price_volatility_scale=PRICE_VOLATILITY_SCALE,
):
    """Return covariance of (Y_W, Y_S) increments over one model step."""
    tau = float(dt_hours)
    if tau <= 0.0:
        raise ValueError("dt_hours must be positive.")
    if math.isclose(LAMBDA_W, LAMBDA_S):
        raise ValueError("LAMBDA_W and LAMBDA_S must differ for the closed form.")

    sigma_w, sigma_s = _effective_sigmas(wind_volatility_scale=wind_volatility_scale)
    lambda_w = float(LAMBDA_W)
    lambda_s = float(LAMBDA_S)
    coupling = float(C_W)

    var_w = sigma_w**2 / (2.0 * lambda_w) * (1.0 - math.exp(-2.0 * lambda_w * tau))
    var_price_direct = (
        sigma_s**2 / (2.0 * lambda_s) * (1.0 - math.exp(-2.0 * lambda_s * tau))
    )

    mixed_decay = 1.0 - math.exp(-(lambda_s + lambda_w) * tau)
    wind_decay_as_price = (
        sigma_w**2 / (2.0 * lambda_s) * (1.0 - math.exp(-2.0 * lambda_s * tau))
    )
    bracket = (
        var_w
        + wind_decay_as_price
        - 2.0 * sigma_w**2 / (lambda_s + lambda_w) * mixed_decay
    )
    coupling_factor = lambda_s * coupling / (lambda_s - lambda_w)

    var_s = var_price_direct + coupling_factor**2 * bracket
    cov_ws = -coupling_factor * (
        var_w - sigma_w**2 / (lambda_s + lambda_w) * mixed_decay
    )

    price_scale = float(price_volatility_scale)
    covariance = np.array(
        [
            [var_w, cov_ws * price_scale],
            [cov_ws * price_scale, var_s * price_scale**2],
        ],
        dtype=float,
    )
    return 0.5 * (covariance + covariance.T)


def transition_statistics(
    current_wind_speed,
    current_price,
    current_hour,
    dt_hours=1.0,
    wind_volatility_scale=WIND_VOLATILITY_SCALE,
    price_volatility_scale=PRICE_VOLATILITY_SCALE,
):
    """Return mean and covariance for (log W_{n+1}, S_{n+1})."""
    tau = float(dt_hours)
    if tau <= 0.0:
        raise ValueError("dt_hours must be positive.")

    current_hour = float(current_hour)
    next_hour = current_hour + tau
    y_w = wind_residual(current_wind_speed, current_hour)
    y_s = residual(current_price, current_hour)

    exp_w = math.exp(-LAMBDA_W * tau)
    exp_s = math.exp(-LAMBDA_S * tau)
    m_y_w = y_w * exp_w
    price_scale = float(price_volatility_scale)
    m_y_s = y_s * exp_s - price_scale * (
        LAMBDA_S * C_W / (LAMBDA_S - LAMBDA_W)
    ) * y_w * (exp_w - exp_s)

    mean = np.array(
        [
            wind_seasonality(next_hour) + m_y_w,
            seasonal_mean(next_hour) + m_y_s,
        ],
        dtype=float,
    )
    covariance = ou_covariance(
        dt_hours=tau,
        wind_volatility_scale=wind_volatility_scale,
        price_volatility_scale=price_volatility_scale,
    )
    return mean, covariance


def expected_next_price(
    current_price,
    current_hour,
    dt_hours=1.0,
    current_wind_speed=None,
):
    """Return E[S(t + dt) | W(t), S(t)]."""
    wind_speed = (
        wind_seasonal_mean(current_hour)
        if current_wind_speed is None
        else float(current_wind_speed)
    )
    mean, _ = transition_statistics(
        wind_speed,
        current_price,
        current_hour,
        dt_hours=dt_hours,
    )
    return float(mean[1])


def price_shock_std(
    dt_hours=1.0,
    wind_volatility_scale=WIND_VOLATILITY_SCALE,
    price_volatility_scale=PRICE_VOLATILITY_SCALE,
):
    """Return std. dev. of the coupled price shock over one model step."""
    covariance = ou_covariance(
        dt_hours=dt_hours,
        wind_volatility_scale=wind_volatility_scale,
        price_volatility_scale=price_volatility_scale,
    )
    return float(math.sqrt(max(covariance[1, 1], 0.0)))


def wind_shock_std(
    dt_hours=1.0,
    wind_volatility_scale=WIND_VOLATILITY_SCALE,
    price_volatility_scale=PRICE_VOLATILITY_SCALE,
):
    """Return std. dev. of the log-wind shock over one model step."""
    covariance = ou_covariance(
        dt_hours=dt_hours,
        wind_volatility_scale=wind_volatility_scale,
        price_volatility_scale=price_volatility_scale,
    )
    return float(math.sqrt(max(covariance[0, 0], 0.0)))


def wind_price_correlation(
    dt_hours=1.0,
    wind_volatility_scale=WIND_VOLATILITY_SCALE,
    price_volatility_scale=PRICE_VOLATILITY_SCALE,
):
    """Return the one-step correlation between log-wind and price shocks."""
    covariance = ou_covariance(
        dt_hours=dt_hours,
        wind_volatility_scale=wind_volatility_scale,
        price_volatility_scale=price_volatility_scale,
    )
    denom = math.sqrt(max(covariance[0, 0] * covariance[1, 1], 0.0))
    return float(covariance[0, 1] / denom) if denom > 0.0 else 0.0


def sample_next_state(
    current_wind_speed,
    current_price,
    current_hour,
    dt_hours=1.0,
    rng=None,
    wind_volatility_scale=WIND_VOLATILITY_SCALE,
    price_volatility_scale=PRICE_VOLATILITY_SCALE,
):
    """Sample the next wind speed and electricity price."""
    generator = np.random.default_rng() if rng is None else rng
    mean, covariance = transition_statistics(
        current_wind_speed,
        current_price,
        current_hour,
        dt_hours=dt_hours,
        wind_volatility_scale=wind_volatility_scale,
        price_volatility_scale=price_volatility_scale,
    )
    log_wind, price = generator.multivariate_normal(mean, covariance)
    return float(math.exp(log_wind)), float(price)


def sample_next_price(
    current_price,
    current_hour,
    dt_hours=1.0,
    rng=None,
    current_wind_speed=None,
):
    """Sample one next price, using seasonal wind if no wind state is supplied."""
    wind_speed = (
        wind_seasonal_mean(current_hour)
        if current_wind_speed is None
        else float(current_wind_speed)
    )
    _, price = sample_next_state(
        wind_speed,
        current_price,
        current_hour,
        dt_hours=dt_hours,
        rng=rng,
    )
    return price


def load_price_model_data(
    num_hours=DEFAULT_PRICE_MODEL_HOURS,
    initial_price=None,
    start_hour=0.0,
    dt_hours=1.0,
    seed=None,
    initial_wind_speed=None,
    return_wind=False,
    wind_volatility_scale=WIND_VOLATILITY_SCALE,
    price_volatility_scale=PRICE_VOLATILITY_SCALE,
):
    """Generate prices from the coupled stochastic wind-price model."""

    rng = np.random.default_rng(seed)
    prices_full = np.empty(num_hours, dtype=float)
    wind_full = np.empty(num_hours, dtype=float)
    current_hour = float(start_hour)

    wind_full[0] = (
        wind_seasonal_mean(current_hour)
        if initial_wind_speed is None
        else float(initial_wind_speed)
    )
    prices_full[0] = (
        seasonal_mean(current_hour) if initial_price is None else float(initial_price)
    )

    for idx in range(1, num_hours):
        wind_full[idx], prices_full[idx] = sample_next_state(
            wind_full[idx - 1],
            prices_full[idx - 1],
            current_hour,
            dt_hours=dt_hours,
            rng=rng,
            wind_volatility_scale=wind_volatility_scale,
            price_volatility_scale=price_volatility_scale,
        )
        current_hour += dt_hours

    global_price_scale = np.percentile(np.abs(prices_full), 95) + 1e-6

    print(f"Generated coupled wind-price dataset length: {len(prices_full)} hours")
    print(f"Global price scale (95th pct): {global_price_scale:.4f}")
    print(
        "Coupled price shock std "
        f"({dt_hours:g} h): "
        f"{price_shock_std(dt_hours, wind_volatility_scale, price_volatility_scale):.4f} EUR/MWh"
    )
    print(
        "Log-wind shock std "
        f"({dt_hours:g} h): "
        f"{wind_shock_std(dt_hours, wind_volatility_scale, price_volatility_scale):.4f}"
    )
    print(
        "Wind-price shock correlation "
        f"({dt_hours:g} h): "
        f"{wind_price_correlation(dt_hours, wind_volatility_scale, price_volatility_scale):.4f}"
    )

    if return_wind:
        return prices_full, wind_full, global_price_scale
    return prices_full, global_price_scale
