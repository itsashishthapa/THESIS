# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:16:08 2025

@author: Loukas Kyriakidis

Optimization of the battery model in Pyomo

"""

import time
import numpy as np
import pandas as pd

# Pyomo 6.7 expects NumPy 1.x scalar aliases when NumPy is already imported.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

from pyomo.environ import *


def optimize_battery_pyomo(prices, P_max=20000, E_max=2e5*3600, SOC_0=0.5, k=500, verbose=True):
    """
    Optimize battery charging/discharging using Pyomo IPOPT solver.
    
    Args:
        prices: List or array of electricity prices (in MWh or base unit)
        P_max: Maximum power (default 20000 W)
        E_max: Battery capacity (default 2e5*3600 Ws)
        SOC_0: Initial state of charge (default 0.5)
        k: Smoothing parameter for tanh approximation (default 500)
        verbose: Print solver output (default True)
    
    Returns:
        Dictionary containing:
            - cost: Optimized total cost (€)
            - time: Optimization time (seconds)
            - u: Control inputs array
            - SOC: State of charge array
            - P: Power array
            - P_actual: Actual power (post-efficiency) array
            - status: 'success' or 'failed'
    """
    try:
        T = len(prices)
        model = ConcreteModel()
        
        # Sets
        model.T = RangeSet(0, T-1)
        price_dict = {t: prices[t]/(1e6) for t in range(T)}
        
        # Parameters
        eta = 0.9  # charge/discharge efficiency
        dt = 3600  # in s
        
        # Variables
        model.u = Var(model.T, bounds=(-1, 1))
        model.SOC = Var(model.T, bounds=(0, 1))
        model.P = Var(model.T, bounds=(-P_max, P_max))
        model.P_actual = Var(model.T, bounds=(-P_max, P_max))
        
        # Objective: Minimize cost from grid
        def objective(m):
            return sum(price_dict[t] * m.P[t] for t in m.T)
        
        # Constraint: P = u * P_max with tanh smoothing
        def power_eq(m, t):
            tanh_term1 = 0.5 * (1 + tanh(k * (m.u[t] - 0.0025)))
            tanh_term2 = 0.5 * (1 + tanh(k * (0.995 - m.SOC[t])))
            tanh_term3 = 0.5 * (1 - tanh(k * (m.u[t] + 0.0025)))
            tanh_term4 = 0.5 * (1 + tanh(k * (m.SOC[t] - 0.005)))
            return m.P[t] == P_max * (tanh_term1 * m.u[t] * tanh_term2 + tanh_term3 * m.u[t] * tanh_term4)
        
        model.power_constraint = Constraint(model.T, rule=power_eq)
        
        # Constraint for P_actual with efficiency
        def p_actual(m, t):
            return m.P_actual[t] == eta * m.P[t]
        
        model.p_actual_constraint = Constraint(model.T, rule=p_actual)
        
        # SOC dynamics
        def soc_dynamics(m, t):
            if t == 0:
                return m.SOC[t] == SOC_0 + dt * m.P_actual[t] / E_max
            else:
                return m.SOC[t] == m.SOC[t-1] + dt * m.P_actual[t] / E_max
        
        model.soc_constraint = Constraint(model.T, rule=soc_dynamics)
        
        # Solve with IPOPT
        model.obj = Objective(rule=objective, sense=minimize)
        solver = SolverFactory('ipopt')
        solver.options['max_iter'] = 1000
        
        start_time = time.time()
        solver.solve(model, tee=verbose)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        obj_value = float(value(model.obj))
        
        # Extract results
        u_opt = [float(value(model.u[t])) for t in range(T)]
        soc_opt = [float(value(model.SOC[t])) for t in range(T)]
        p_opt = [float(value(model.P[t])) for t in range(T)]
        p_actual_opt = [float(value(model.P_actual[t])) for t in range(T)]
        
        return {
            'cost': obj_value,
            'time': elapsed_time,
            'u': np.array(u_opt),
            'SOC': np.array(soc_opt),
            'P': np.array(p_opt),
            'P_actual': np.array(p_actual_opt),
            'status': 'success'
        }
    
    except Exception as e:
        if verbose:
            print(f"Pyomo optimization failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def load_price_data():
    """Load electricity price data from file"""
    df = pd.read_csv("input_data/electricity_price.txt", sep='\\s+', header=None)
    prices = df[1].tolist()
    return prices


if __name__ == '__main__':
    # Load price data
    prices_full = load_price_data()
    
    # Get prices for a specific window (e.g., hours 72-96)
    prices = prices_full[72:96]
    
    # Run optimization
    result = optimize_battery_pyomo(prices, verbose=True)
    
    if result['status'] == 'success':
        print("\nOptimization successful!")
        print("Objective value:", round(result['cost'], 2), "Euro")
        print(f"Optimization time: {result['time']:.2f} seconds")
    else:
        print("Optimization failed:", result.get('error', 'Unknown error'))
