from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

try:
    from pyomo.environ import (
        ConcreteModel,
        Constraint,
        ConstraintList,
        NonNegativeReals,
        Objective,
        Param,
        RangeSet,
        Reals,
        SolverFactory,
        SolverStatus,
        TerminationCondition,
        Var,
        minimize,
        value,
    )
except ImportError as error:
    PYOMO_IMPORT_ERROR = error
else:
    PYOMO_IMPORT_ERROR = None


DEFAULT_DATA_PATH = Path(__file__).with_name("WKA_Pe_merged.csv")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name(
    "Results_Paper_hybrid_keksfabrik_24h_multistarts_ipopt_new.xlsx"
)


def _require_pyomo() -> None:
    if PYOMO_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Pyomo is required. Install it in the active environment before running "
            "the Cookie Factory optimization."
        ) from PYOMO_IMPORT_ERROR


def build_cookie_model(
    power_price: np.ndarray | list[float],
    power_wka: np.ndarray | list[float],
):
    """Build the Cookie Factory Pyomo model for one price and wind window."""
    _require_pyomo()

    power_price = np.asarray(power_price, dtype=float)
    power_wka = np.asarray(power_wka, dtype=float)
    if power_price.ndim != 1 or power_wka.ndim != 1:
        raise ValueError("power_price and power_wka must be one-dimensional.")
    if len(power_price) == 0 or len(power_price) != len(power_wka):
        raise ValueError("power_price and power_wka must be non-empty and equally sized.")

    model = ConcreteModel()
    model.Time = RangeSet(0, len(power_price) - 1)

    eps = 1.0e-6
    eff_ch = 0.9
    eff_dch = 0.9
    storage_initial_temperature = 250.0
    storage_mass = 600000.0
    storage_heat_capacity = 1.025
    fluid_heat_capacity = 2.21
    delta_t = 3600.0

    def random_initializer(lower: float, upper: float):
        def initializer(model_instance, time_index):
            return random.uniform(lower, upper)

        return initializer

    model.Pe = Param(model.Time, initialize=dict(enumerate(power_price)))
    model.WKA = Param(model.Time, initialize=dict(enumerate(power_wka)))

    model.Thx1 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(177, 250),
        initialize=random_initializer(177, 250),
    )
    model.Thx2 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(75, 75),
        initialize=75.0,
    )
    model.M1 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(5, 16),
        initialize=random_initializer(5, 16),
    )
    model.P = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(0, 5000),
        initialize=random_initializer(0, 5000),
    )
    model.N = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(0.8, 1.53),
        initialize=random_initializer(0.8, 1.53),
    )
    model.x1 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=random_initializer(0, 1),
    )
    model.x2 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(0, 1),
        initialize=random_initializer(0, 1),
    )
    model.Qch = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(0, 5000),
        initialize=random_initializer(0, 5000),
    )
    model.Qdch = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(0, 5000),
        initialize=random_initializer(0, 5000),
    )
    model.T1Out = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(239, 400),
        initialize=random_initializer(239, 400),
    )
    model.T2Out = Var(
        model.Time,
        domain=Reals,
        bounds=(-60, 90),
        initialize=random_initializer(-60, 90),
    )
    model.T3 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(239, 324),
        initialize=random_initializer(239, 324),
    )
    model.T4 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(183, 193),
        initialize=random_initializer(183, 193),
    )
    model.S = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(183, 324),
        initialize=random_initializer(183, 324),
    )
    model.T2 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(183, 332),
        initialize=random_initializer(183, 332),
    )
    model.T5 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(183, 332),
        initialize=random_initializer(183, 332),
    )
    model.T0 = Var(
        model.Time,
        domain=NonNegativeReals,
        bounds=(183, 324),
        initialize=random_initializer(183, 324),
    )

    model.obj = Objective(
        expr=sum(model.Pe[t] / 1000.0 * model.P[t] for t in model.Time),
        sense=minimize,
    )

    def f1_rule(m, t):
        return m.T1Out[t] == (
            95.9612 + 0.93433 * m.Thx1[t] - 0.327753 * m.M1[t]
            + 0.0146542 * m.Thx2[t] - 271.354 * m.N[t]
            + 0.00104853 * m.Thx1[t] ** 2 + 0.0211819 * m.Thx1[t] * m.M1[t]
            - 0.706122 * m.Thx1[t] * m.N[t] + 1.04924 * m.M1[t] ** 2
            - 0.00388073 * m.M1[t] * m.Thx2[t] - 29.4801 * m.M1[t] * m.N[t]
            + 0.0595068 * m.Thx2[t] * m.N[t] + 562.428 * m.N[t] ** 2
            - 0.000716825 * m.Thx1[t] ** 2 * m.N[t]
            - 0.00148575 * m.Thx1[t] * m.M1[t] ** 2
            + 0.0229386 * m.Thx1[t] * m.M1[t] * m.N[t]
            + 0.203578 * m.Thx1[t] * m.N[t] ** 2 - 0.0405702 * m.M1[t] ** 3
            + 0.881391 * m.M1[t] ** 2 * m.N[t] - 2.18172 * m.M1[t] * m.N[t] ** 2
            - 151.476 * m.N[t] ** 3
        )

    def f2_rule(m, t):
        return m.T2Out[t] == (
            93.3958 - 0.00692483 * m.Thx1[t] - 0.770173 * m.M1[t]
            + 1.30277 * m.Thx2[t] - 183.866 * m.N[t]
            + 0.00313225 * m.Thx1[t] * m.M1[t] + 0.234082 * m.Thx1[t] * m.N[t]
            + 0.106964 * m.M1[t] ** 2 - 2.34999 * m.M1[t] * m.N[t]
            - 0.555879 * m.Thx2[t] * m.N[t] + 30.2955 * m.N[t] ** 2
        )

    def f3_rule(m, t):
        return m.P[t] + m.WKA[t] == 3 * (
            127.87 + 2.06342 * m.Thx1[t] + 2.55723 * m.M1[t]
            + 0.756419 * m.Thx2[t] - 1164.84 * m.N[t]
            - 0.0168942 * m.Thx1[t] * m.M1[t] - 2.60579 * m.Thx1[t] * m.N[t]
            - 0.540713 * m.M1[t] ** 2 + 13.3204 * m.M1[t] * m.N[t]
            - 1.3829 * m.Thx2[t] * m.N[t] + 1556.66 * m.N[t] ** 2
        )

    def f4_rule(m, t):
        return m.T3[t] == m.T2[t] * (1 - m.x1[t]) + m.T1Out[t] * m.x1[t]

    def f5_rule(m, t):
        return m.Thx1[t] == m.T4[t] * m.x2[t] + m.T5[t] * (1 - m.x2[t])

    def f6_rule(m, t):
        return m.Qch[t] == (
            (m.T1Out[t] - m.T2[t]) * m.M1[t] * 3 * fluid_heat_capacity * (1 - m.x1[t])
        )

    def f7_rule(m, t):
        return m.Qdch[t] == (
            (m.T5[t] - m.T4[t]) * m.M1[t] * 3 * fluid_heat_capacity * (1 - m.x2[t])
        )

    def f8_rule(m, t):
        return m.T2[t] == m.T1Out[t] - eff_ch * (m.T1Out[t] - m.T0[t])

    def f9_rule(m, t):
        return m.T5[t] == m.T4[t] - eff_dch * (m.T4[t] - m.T0[t])

    def f10_rule(m, t):
        return m.x1[t] + m.x2[t] >= 1

    def f11_rule(m, t):
        return m.Qch[t] * m.Qdch[t] <= eps

    def f12_rule(m, t):
        return m.T4[t] == -188.403 / (m.M1[t] * 3) + 196.3

    def f13_rule(m, t):
        return m.T3[t] == 201.915 + 1819.32 / (m.M1[t] * 3)

    def f14_rule(m, t):
        if t == m.Time.first():
            return m.T0[t] == storage_initial_temperature
        return m.T0[t] == m.S[t - 1]

    def storage_rule(m, t):
        previous_storage = storage_initial_temperature if t == m.Time.first() else m.S[t - 1]
        return m.S[t] == previous_storage + delta_t * (m.Qch[t] - m.Qdch[t]) / (
            storage_mass * storage_heat_capacity
        )

    model.F1 = Constraint(model.Time, rule=f1_rule)
    model.F2 = Constraint(model.Time, rule=f2_rule)
    model.F3 = Constraint(model.Time, rule=f3_rule)
    model.F4 = Constraint(model.Time, rule=f4_rule)
    model.F5 = Constraint(model.Time, rule=f5_rule)
    model.F6 = Constraint(model.Time, rule=f6_rule)
    model.F7 = Constraint(model.Time, rule=f7_rule)
    model.F8 = Constraint(model.Time, rule=f8_rule)
    model.F9 = Constraint(model.Time, rule=f9_rule)
    model.F10 = Constraint(model.Time, rule=f10_rule)
    model.F11 = Constraint(model.Time, rule=f11_rule)
    model.F12 = Constraint(model.Time, rule=f12_rule)
    model.F13 = Constraint(model.Time, rule=f13_rule)
    model.F14 = Constraint(model.Time, rule=f14_rule)
    model.end_storage = ConstraintList()
    model.end_storage.add(model.S[model.Time.last()] == storage_initial_temperature)
    model.storage = Constraint(model.Time, rule=storage_rule)

    return model


def _trajectory_frame(model) -> pd.DataFrame:
    rows = []
    for t in model.Time:
        rows.append(
            {
                "k": int(t),
                "P_grid": value(model.P[t]),
                "T_I": value(model.Thx1[t]),
                "T_III": value(model.Thx2[t]),
                "m_I": value(model.M1[t]),
                "R": value(model.N[t]),
                "T_s": value(model.S[t]),
                "Q_ch": value(model.Qch[t]),
                "Q_dch": value(model.Qdch[t]),
                "T_II": value(model.T1Out[t]),
                "T_IV": value(model.T2Out[t]),
                "T_1": value(model.T2[t]),
                "T_4": value(model.T5[t]),
                "T_0": value(model.T0[t]),
                "beta1": value(model.x1[t]),
                "beta2": value(model.x2[t]),
                "T_2": value(model.T3[t]),
                "T_3": value(model.T4[t]),
            }
        )
    return pd.DataFrame(rows)


def get_cookie_model(
    power_price: np.ndarray | list[float],
    power_wka: np.ndarray | list[float],
    solver_name: str = "ipopt",
    max_iter: int = 1000,
    verbose: bool = True,
) -> dict:
    """Optimize one Cookie Factory window and return Battery-model-style results."""
    try:
        model = build_cookie_model(power_price, power_wka)
        solver = SolverFactory(solver_name)
        if not solver.available(exception_flag=False):
            raise RuntimeError(f"Solver '{solver_name}' is not available on PATH.")
        solver.options["max_iter"] = int(max_iter)

        start_time = time.perf_counter()
        result = solver.solve(model, tee=verbose)
        elapsed_time = time.perf_counter() - start_time
        is_optimal = (
            result.solver.status == SolverStatus.ok
            and result.solver.termination_condition
            in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)
        )
        if not is_optimal:
            return {
                "status": "failed",
                "error": str(result.solver.termination_condition),
                "time": elapsed_time,
            }

        trajectory = _trajectory_frame(model)
        return {
            "status": "success",
            "cost": float(value(model.obj)),
            "time": elapsed_time,
            "trajectory": trajectory,
            "P": trajectory["P_grid"].to_numpy(),
            "T_I": trajectory["T_I"].to_numpy(),
            "m_I": trajectory["m_I"].to_numpy(),
            "R": trajectory["R"].to_numpy(),
            "T_s": trajectory["T_s"].to_numpy(),
            "Q_ch": trajectory["Q_ch"].to_numpy(),
            "Q_dch": trajectory["Q_dch"].to_numpy(),
            "beta1": trajectory["beta1"].to_numpy(),
            "beta2": trajectory["beta2"].to_numpy(),
        }
    except Exception as error:
        if verbose:
            print(f"Cookie Factory optimization failed: {error}")
        return {"status": "failed", "error": str(error)}


def load_cookie_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    price_col: str = "Pe",
    wind_col: str = "WKA",
    hours: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a fixed Cookie Factory price and wind optimization window."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if hours <= 0:
        raise ValueError("hours must be positive.")

    frame = pd.read_csv(data_path, nrows=hours)
    if price_col not in frame.columns or wind_col not in frame.columns:
        raise ValueError(f"Data must include '{price_col}' and '{wind_col}' columns.")
    if len(frame) < hours:
        raise ValueError(f"Data contains {len(frame)} rows, but {hours} are required.")

    return (
        pd.to_numeric(frame[price_col], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(frame[wind_col], errors="raise").to_numpy(dtype=float),
    )


def run_multistart(
    power_price: np.ndarray,
    power_wka: np.ndarray,
    starts: int = 10,
    solver_name: str = "ipopt",
    max_iter: int = 1000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict | None]:
    """Run multiple random initializations and return their summary and best result."""
    if starts <= 0:
        raise ValueError("starts must be positive.")

    summaries = []
    best_result = None
    for start_index in range(starts):
        if verbose:
            print(f"\nStart {start_index + 1}/{starts}")
        result = get_cookie_model(
            power_price,
            power_wka,
            solver_name=solver_name,
            max_iter=max_iter,
            verbose=verbose,
        )
        summaries.append(
            {
                "start": start_index + 1,
                "status": result["status"],
                "cost": result.get("cost", np.nan),
                "solver_time_s": result.get("time", np.nan),
                "error": result.get("error", ""),
            }
        )
        if result["status"] == "success" and (
            best_result is None or result["cost"] < best_result["cost"]
        ):
            best_result = result

    return pd.DataFrame(summaries), best_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the Cookie Factory Pyomo model")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Input CSV path")
    parser.add_argument("--price-col", default="Pe", help="Grid-price column name")
    parser.add_argument("--wind-col", default="WKA", help="Wind-power column name")
    parser.add_argument("--hours", type=int, default=24, help="Optimization horizon")
    parser.add_argument("--starts", type=int, default=10, help="Number of random starts")
    parser.add_argument("--solver", default="ipopt", help="Pyomo solver executable name")
    parser.add_argument("--max-iter", type=int, default=1000, help="Solver iteration limit")
    parser.add_argument("--seed", type=int, default=None, help="Optional random-start seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress solver output")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output Excel path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    power_price, power_wka = load_cookie_data(
        data_path=args.data,
        price_col=args.price_col,
        wind_col=args.wind_col,
        hours=args.hours,
    )
    print("price",power_price, power_wka)
    summary, best_result = run_multistart(
        power_price,
        power_wka,
        starts=args.starts,
        solver_name=args.solver,
        max_iter=args.max_iter,
        verbose=not args.quiet,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="multistart_summary", index=False)
        if best_result is not None:
            best_result["trajectory"].to_excel(writer, sheet_name="best_trajectory", index=False)

    print(f"\nResults written to {output_path}")
    if best_result is None:
        raise SystemExit("No optimization start reached an optimal solution.")
    print(f"Best objective: {best_result['cost']:.6f}")
    print(f"Best solver time: {best_result['time']:.3f} s")


if __name__ == "__main__":
    main()
