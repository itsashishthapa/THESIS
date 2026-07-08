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

from __future__ import annotations

import numpy as np


LAM_S = 0.0535257810990052
SIG_S_2 = 17.687548961803607

LAM_W = 0.1444180172085898
SIG_W_2 = 0.08396670351568147

CW = 2.382279201991361
DEFAULT_DT_HOURS = 1.0
DEFAULT_PRICE_MODEL_HOURS = 8784


def _return_scalar_if_scalar(value, reference):
    return float(value) if np.ndim(np.asarray(reference, dtype=float)) == 0 else value


def var_y_w(t_n: float, t: float) -> float:
    return (SIG_W_2 / (2.0 * LAM_W)) * (1.0 - np.exp(-2.0 * LAM_W * (t - t_n)))


def var_w(t_n: float, t: float) -> float:
    return var_y_w(t_n, t)


def var_y_s(t_n: float, t: float) -> float:
    return (SIG_S_2 / (2.0 * LAM_S)) * (1.0 - np.exp(-2.0 * LAM_S * (t - t_n)))


def var_s(t_n: float, t: float) -> float:
    sig_y_s = var_y_s(t_n, t)
    sig_y_w = var_y_w(t_n, t)
    return sig_y_s + (((LAM_S * CW) ** 2) / ((LAM_S - LAM_W) ** 2)) * (
        sig_y_w
        + (SIG_W_2 / SIG_S_2) * sig_y_s
        - ((2.0 * SIG_W_2) / (LAM_S + LAM_W))
        * (1.0 - np.exp(-(LAM_S + LAM_W) * (t - t_n)))
    )


def covariance_ws(t_n: float, t: float) -> float:
    sig_y_w = var_y_w(t_n, t)
    return (-(LAM_S * CW) / (LAM_S - LAM_W)) * (
        sig_y_w
        - (SIG_W_2 / (LAM_W + LAM_S))
        * (1.0 - np.exp(-(LAM_S + LAM_W) * (t - t_n)))
    )


def _step_statistics(dt_hours: float) -> tuple[float, float, float, float]:
    sigma_2_w = var_w(0.0, dt_hours)
    sigma_2_s = var_s(0.0, dt_hours)
    sigma_2_ws = covariance_ws(0.0, dt_hours)
    denom = np.sqrt(sigma_2_w) * np.sqrt(sigma_2_s)
    rho = sigma_2_ws / denom if denom > 0.0 else 0.0
    return float(sigma_2_w), float(sigma_2_s), float(sigma_2_ws), float(rho)


SIGMA_2_W, SIGMA_2_S, SIGMA_2_WS, RHO = _step_statistics(DEFAULT_DT_HOURS)


def g_w(x):
    a = 0.13574545681905442
    b = -0.3286669816831083
    c = 1.6496304557720327
    t1 = 1034.526102455241
    t2 = 1.1707259416700286

    hour = np.asarray(x, dtype=float)
    value = (
        a * np.cos(2.0 * np.pi * (hour - t1) / (365.0 * 24.0))
        + b * np.cos(2.0 * np.pi * (hour - t2) / 24.0)
        + c
    )
    return _return_scalar_if_scalar(value, x)


def g_s(x):
    a = -11.203811091724104
    b = 4.257077726408406
    c = -6.664242994338143
    d = 30.494543670242404
    t1 = -14782.502553513012
    t2 = -6.7823380716985815
    t3 = -9.501631670758995

    hour = np.asarray(x, dtype=float)
    value = (
        a * np.cos(2.0 * np.pi * (hour - t1) / (365.0 * 24.0))
        + b * np.cos(2.0 * np.pi * (hour - t2) / 24.0)
        + c * np.cos(2.0 * np.pi * (hour - t3) / 12.0)
        + d
    )
    return _return_scalar_if_scalar(value, x)


def wind_seasonality(hour):
    """Return the deterministic log-wind seasonality g_W(t)."""
    return g_w(hour)


def price_seasonality(hour):
    """Return the deterministic electricity-price seasonality g_S(t)."""
    return g_s(hour)


def seasonality(hour):
    """Compatibility alias for the deterministic price seasonality."""
    return price_seasonality(hour)


def wind_seasonal_mean(hour):
    """Return the seasonal mean wind speed exp(g_W(t))."""
    h = np.asarray(hour, dtype=float)
    value = np.exp(g_w(h))
    return _return_scalar_if_scalar(value, hour)


def price_residual(price, hour):
    """Return the price residual Y_S(t) = S(t) - g_S(t)."""
    return float(price - price_seasonality(hour))


def wind_residual(wind_speed, hour):
    """Return the log-wind residual Y_W(t) = log(W(t)) - g_W(t)."""
    return float(np.log(max(float(wind_speed), 1e-12)) - wind_seasonality(hour))


def wind_price_correlation(dt_hours: float = DEFAULT_DT_HOURS) -> float:
    """Return the one-step correlation between log-wind and price shocks."""
    return _step_statistics(dt_hours)[3]


def wind_shock_std(dt_hours: float = DEFAULT_DT_HOURS) -> float:
    """Return the standard deviation of the log-wind shock."""
    return float(np.sqrt(max(var_w(0.0, dt_hours), 0.0)))


def price_shock_std(dt_hours: float = DEFAULT_DT_HOURS) -> float:
    """Return the standard deviation of the price shock."""
    return float(np.sqrt(max(var_s(0.0, dt_hours), 0.0)))


def transition(
    n: float,
    yw: float,
    ys: float,
    rng: np.random.Generator,
    dt_hours: float = DEFAULT_DT_HOURS,
) -> tuple[float, float, float, float]:
    """Sample one transition of the coupled wind-price process."""
    sigma_2_w, sigma_2_s, _, rho = _step_statistics(dt_hours)
    b = rng.normal(0.0, 1.0, size=2)
    next_hour = n + dt_hours
    exp_w = np.exp(-LAM_W * dt_hours)
    exp_s = np.exp(-LAM_S * dt_hours)

    yw_new = g_w(next_hour) + yw * exp_w + np.sqrt(sigma_2_w) * b[0]
    ys_new = (
        ys * exp_s
        - ((LAM_S * CW * yw) / (LAM_S - LAM_W)) * (exp_w - exp_s)
        + np.sqrt(sigma_2_s) * (rho * b[0] + np.sqrt(max(1.0 - rho**2, 0.0)) * b[1])
    )
    w_new = np.exp(yw_new)
    s_new = ys_new + g_s(next_hour)
    return float(yw_new), float(w_new), float(ys_new), float(s_new)


def simulate(
    hours: int,
    seed: int | None = None,
    start_hour: float = 0.0,
    dt_hours: float = DEFAULT_DT_HOURS,
    initial_price: float | None = None,
    initial_wind_speed: float | None = None,
) -> dict[str, np.ndarray]:
    """Generate one coupled wind-price path."""
    if hours < 0:
        raise ValueError("hours must be non-negative.")
    if dt_hours <= 0.0:
        raise ValueError("dt_hours must be positive.")

    rng = np.random.default_rng(seed)
    hour = start_hour + np.arange(hours + 1, dtype=float) * dt_hours

    yw = np.zeros(hours + 1)
    wind = np.zeros(hours + 1)
    ys = np.zeros(hours + 1)
    price = np.zeros(hours + 1)

    wind[0] = (
        wind_seasonal_mean(start_hour)
        if initial_wind_speed is None
        else float(initial_wind_speed)
    )
    yw[0] = wind_residual(wind[0], start_hour)
    price[0] = (
        price_seasonality(start_hour) if initial_price is None else float(initial_price)
    )
    ys[0] = price_residual(price[0], start_hour)

    for idx in range(hours):
        yw[idx + 1], wind[idx + 1], ys[idx + 1], price[idx + 1] = transition(
            hour[idx],
            yw[idx],
            ys[idx],
            rng,
            dt_hours=dt_hours,
        )

    return {
        "hour": hour,
        "yw": yw,
        "wind": wind,
        "ys": ys,
        "price": price,
        "g_w": g_w(hour),
        "g_s": g_s(hour),
    }


def load_price_model_data(
    num_hours=DEFAULT_PRICE_MODEL_HOURS,
    initial_price=None,
    start_hour=0.0,
    dt_hours=DEFAULT_DT_HOURS,
    seed=None,
    initial_wind_speed=None,
    return_wind=False,
):
    """Generate prices from the coupled stochastic wind-price process."""
    if num_hours <= 0:
        raise ValueError("num_hours must be positive.")
    if dt_hours <= 0.0:
        raise ValueError("dt_hours must be positive.")

    path = simulate(
        num_hours - 1,
        seed=seed,
        start_hour=start_hour,
        dt_hours=dt_hours,
        initial_price=initial_price,
        initial_wind_speed=initial_wind_speed,
    )
    prices_full = path["price"]
    wind_full = path["wind"]
    global_price_scale = np.percentile(np.abs(prices_full), 95) + 1e-6

    print(f"Generated coupled wind-price dataset length: {len(prices_full)} hours")
    print(f"Global price scale (95th pct): {global_price_scale:.4f}")
    print(
        "Coupled price shock std "
        f"({dt_hours:g} h): {price_shock_std(dt_hours):.4f} EUR/MWh"
    )
    print(f"Log-wind shock std ({dt_hours:g} h): {wind_shock_std(dt_hours):.4f}")
    print(
        "Wind-price shock correlation "
        f"({dt_hours:g} h): {wind_price_correlation(dt_hours):.4f}"
    )

    if return_wind:
        return prices_full, wind_full, global_price_scale
    return prices_full, global_price_scale
