"""Coupled wind and electricity price process used as the price model."""

from __future__ import annotations

import numpy as np


LAM_S = 0.0535257810990052
SIG_S_2 = 17.687548961803607

LAM_W = 0.1444180172085898
SIG_W_2 = 0.08396670351568147

CW = 2.382279201991361
DELTA_T = 1.0
DEFAULT_PRICE_MODEL_HOURS = 8784


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


SIGMA_2_W = var_w(0.0, DELTA_T)
SIGMA_2_S = var_s(0.0, DELTA_T)
SIGMA_2_WS = covariance_ws(0.0, DELTA_T)
RHO = SIGMA_2_WS / (np.sqrt(SIGMA_2_W) * np.sqrt(SIGMA_2_S))


def g_w(x: np.ndarray | float) -> np.ndarray | float:
    a = 0.13574545681905442
    b = -0.3286669816831083
    c = 1.6496304557720327
    t1 = 1034.526102455241
    t2 = 1.1707259416700286

    return (
        a * np.cos(2.0 * np.pi * (x - t1) / (365.0 * 24.0))
        + b * np.cos(2.0 * np.pi * (x - t2) / 24.0)
        + c
    )


def g_s(x: np.ndarray | float) -> np.ndarray | float:
    a = -11.203811091724104
    b = 4.257077726408406
    c = -6.664242994338143
    d = 30.494543670242404
    t1 = -14782.502553513012
    t2 = -6.7823380716985815
    t3 = -9.501631670758995

    return (
        a * np.cos(2.0 * np.pi * (x - t1) / (365.0 * 24.0))
        + b * np.cos(2.0 * np.pi * (x - t2) / 24.0)
        + c * np.cos(2.0 * np.pi * (x - t3) / 12.0)
        + d
    )


def transition(
    n: int,
    yw: float,
    ys: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    b = rng.normal(0.0, 1.0, size=2)
    yw_new = (
        g_w(n + 1)
        + yw * np.exp(-LAM_W * DELTA_T)
        + np.sqrt(SIGMA_2_W) * b[0]
    )
    ys_new = (
        ys * np.exp(-LAM_S * DELTA_T)
        - ((LAM_S * CW * yw) / (LAM_S - LAM_W))
        * (np.exp(-LAM_W * DELTA_T) - np.exp(-LAM_S * DELTA_T))
        + np.sqrt(SIGMA_2_S) * (RHO * b[0] + np.sqrt(1.0 - RHO**2) * b[1])
    )
    w_new = np.exp(yw_new)
    s_new = ys_new + g_s(n + 1)
    return yw_new, w_new, ys_new, s_new


def simulate(hours: int, seed: int | None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hour = np.arange(hours + 1)

    yw = np.zeros(hours + 1)
    wind = np.zeros(hours + 1)
    ys = np.zeros(hours + 1)
    price = np.zeros(hours + 1)

    wind[0] = np.exp(yw[0])
    price[0] = ys[0] + g_s(0)

    for n in range(hours):
        yw[n + 1], wind[n + 1], ys[n + 1], price[n + 1] = transition(
            n,
            yw[n],
            ys[n],
            rng,
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
    dt_hours=DELTA_T,
    seed=None,
):
    if num_hours <= 0:
        raise ValueError("num_hours must be positive.")
    if initial_price is not None:
        raise ValueError("initial_price is not supported by the coupled plot process.")
    if start_hour != 0.0:
        raise ValueError("start_hour is not supported by the coupled plot process.")
    if dt_hours != DELTA_T:
        raise ValueError("dt_hours must be 1.0 for the coupled plot process.")

    path = simulate(num_hours - 1, seed)
    prices_full = path["price"]
    global_price_scale = np.percentile(np.abs(prices_full), 95) + 1e-6

    print(f"Generated coupled wind-price dataset length: {len(prices_full)} hours")
    print(f"Global price scale (95th pct): {global_price_scale:.4f}")

    return prices_full, global_price_scale
