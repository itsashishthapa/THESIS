"""
Electricity-only price model from Pilling et al. (2025).

This file uses the constants reported in the paper for the electricity price
seasonality and Ornstein-Uhlenbeck residual. It does not calibrate parameters
from data, and it intentionally omits the wind process and wind-price
correlation term from the full paper model.

The calibrated OU residual volatility in the paper is very small compared with
the seasonal price component. The module-level constants below apply a fixed
scenario volatility scale and level shift for generated price paths.

Time is measured in hours. Prices are in EUR/MWh.
"""

import math

import numpy as np


# Seasonality constants for mu_S(t), Table G.1.
K0_S = -11.2038
K1_S = 4.2571
K2_S = -6.6642
K3_S = 30.4945
T1_S = -14782.5
T2_S = -6.7823
T3_S = -9.5016

YEARLY_PERIOD_HOURS = 8760.0
DAILY_PERIOD_HOURS = 24.0
HALF_DAILY_PERIOD_HOURS = 12.0
DEFAULT_PRICE_MODEL_HOURS = 8784
RESIDUAL_VOLATILITY_SCALE = 50.0
PRICE_LEVEL_SHIFT = 30.0

# Ornstein-Uhlenbeck constants for the electricity price residual, Table B.1.
LAMBDA_S = 0.2534
SIGMA_S = 0.1072


def seasonality(hour):
    """Return the deterministic seasonal price component mu_S(t)."""
    h = np.asarray(hour, dtype=float)
    two_pi = 2.0 * np.pi
    value = (
        K0_S
        + K1_S * np.cos(two_pi * (h - T1_S) / YEARLY_PERIOD_HOURS)
        + K2_S * np.cos(two_pi * (h - T2_S) / DAILY_PERIOD_HOURS)
        + K3_S * np.cos(two_pi * (h - T3_S) / HALF_DAILY_PERIOD_HOURS)
    )
    return float(value) if np.ndim(h) == 0 else value


def seasonal_mean(hour):
    """Return the seasonal mean price used by the simulated process."""
    value = seasonality(hour) + PRICE_LEVEL_SHIFT
    return float(value) if np.ndim(np.asarray(hour, dtype=float)) == 0 else value


def residual(price, hour):
    """Return the price residual Y_S(t) = S(t) - mu_S(t)."""
    return float(price - seasonal_mean(hour))


def expected_next_price(
    current_price,
    current_hour,
    dt_hours=1.0,
):
    """Return the mean next price E[S(t + dt) | S(t)]."""
    if dt_hours <= 0.0:
        raise ValueError("dt_hours must be positive.")

    current_residual = residual(current_price, current_hour)
    next_residual = math.exp(-LAMBDA_S * dt_hours) * current_residual
    return float(seasonal_mean(current_hour + dt_hours) + next_residual)


def price_shock_std(dt_hours=1.0):
    """Return the standard deviation of the OU price shock over dt_hours."""
    if dt_hours <= 0.0:
        raise ValueError("dt_hours must be positive.")

    variance = (
        SIGMA_S**2
        / (2.0 * LAMBDA_S)
        * (1.0 - math.exp(-2.0 * LAMBDA_S * dt_hours))
    )
    return RESIDUAL_VOLATILITY_SCALE * math.sqrt(max(variance, 0.0))


def sample_next_price(
    current_price,
    current_hour,
    dt_hours=1.0,
    rng=None,
):
    """Sample one next price from the electricity price process."""
    generator = np.random.default_rng() if rng is None else rng
    mean = expected_next_price(
        current_price,
        current_hour,
        dt_hours,
    )
    shock = price_shock_std(dt_hours) * generator.standard_normal()
    return float(mean + shock)


def load_price_model_data(
    num_hours=DEFAULT_PRICE_MODEL_HOURS,
    initial_price=None,
    start_hour=0.0,
    dt_hours=1.0,
    seed=None,
):
    """Generate electricity price data using the stochastic price model."""
    if num_hours <= 0:
        raise ValueError("num_hours must be positive.")
    if dt_hours <= 0.0:
        raise ValueError("dt_hours must be positive.")

    rng = np.random.default_rng(seed)
    prices_full = np.empty(num_hours, dtype=float)
    prices_full[0] = (
        seasonal_mean(start_hour) if initial_price is None else float(initial_price)
    )

    current_hour = float(start_hour)
    for idx in range(1, num_hours):
        prices_full[idx] = sample_next_price(
            prices_full[idx - 1],
            current_hour,
            dt_hours=dt_hours,
            rng=rng,
        )
        current_hour += dt_hours

    global_price_scale = np.percentile(np.abs(prices_full), 95) + 1e-6

    print(f"Generated price-model dataset length: {len(prices_full)} hours")
    print(f"Global price scale (95th pct): {global_price_scale:.4f}")
    print(
        "Price-model residual shock std "
        f"({dt_hours:g} h): {price_shock_std(dt_hours):.4f} EUR/MWh"
    )

    return prices_full, global_price_scale
