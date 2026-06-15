# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:52:23 2025

@author: bubl_sa
"""
# =============================================================================
# # Imports:
# =============================================================================
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import *


# =============================================================================
# # Methods:
# =============================================================================
def get_battery_model(e_price):
    """
    Parameters
    ----------
    e_price : list
        electricity prices in €/kWh

    Returns
    -------
    pyomo.core.base.PyomoModel.ConcreteModel
        mathematical model instance according to pyomo logic

    """
    T = T = len(e_price)

    model = ConcreteModel()

    # sets
    model.T = RangeSet(0, T-1)
    price_dict = {t: e_price[t] for t in range(T)}

    # Parameters
    P_max = 20.0 # in kW
    eta = 0.9  # charge/discharge efficiency
    E_max = 2.0e2 # in Ws
    dt = 1.0 # in h
    SOC_0 = 0.5 # initial SOC
    k=700 # paramter for smoothing
    epsilon=1.0 # parameter for smoothing
    
    # variables
    model.u = Var(model.T, bounds=(-1.0, 1.0))
    model.SOC = Var(model.T, bounds=(0, 1))
    model.SOC[len(model.SOC)-1].bounds=(0.5,1.0) # endpoint constraint
    model.P = Var(model.T, bounds=(-20.0, 20.0))
    model.P_actual = Var(model.T, bounds=(-20.0, 20.0))
    model.acc_cost = Var(model.T, bounds=(-100.0, 100.0))
    model.arg = Var(model.T, bounds=(-1000000.0, 1000000.0))

    # Objective: Minimize cost from grid
    def objective(m):
        return sum(price_dict[t] * m.P[t] * dt for t in m.T)
    model.obj = Objective(rule=objective, sense=minimize)
    
    # Constraint
    def acc_cost_eq(m,t):
        if t == 0:
            return 0.0 == m.acc_cost[0] - ((price_dict[0] * m.P[0] *dt))
        else:
            return 0.0 == m.acc_cost[t] - (m.acc_cost[t-1] + (price_dict[t] * m.P[t] *dt))
    model.acc_cost_constraint = Constraint(model.T, rule=acc_cost_eq)

    # Constraint: P = u * P_max
    def power_eq(m, t):
        return 0.0 == m.P[t] - ((P_max*(0.5*(1+tanh(k*(m.u[t]-0.0025))) * m.u[t] * 0.5 * (1 + tanh(k * (0.995-m.SOC[t]))) +\
                         0.5*(1-tanh(k*(m.u[t]+0.0025)))*m.u[t]*0.5*(1+tanh(k*(m.SOC[t]-0.005))))))
    model.power_constraint = Constraint(model.T, rule=power_eq)
    
    # Auxilary variable
    def arg_aux_eq(m, t):
        return 0.0 == m.arg[t] - (m.P[t]/epsilon)
    model.arg_aux_constraint = Constraint(model.T, rule=arg_aux_eq)

    # Constraint for P_actual 
    def p_actual(m, t):
        return 0.0 == m.P_actual[t]  - (m.P[t] * 0.5* ( (1 + tanh(m.arg[t])) * eta + (1 - tanh(m.arg[t]))/eta))
    
    model.p_actual_constraint = Constraint(model.T, rule=p_actual)
    
    # SOC dynamics
    def soc_dynamics(m, t):
        if t == 0:
            return 0.0 == m.SOC[t] - (SOC_0 + dt * m.P_actual[t]/E_max)
        else:
            return 0.0 == m.SOC[t] - (m.SOC[t-1] + dt * m.P_actual[t]/E_max)
    model.soc_constraint = Constraint(model.T, rule=soc_dynamics)

    return model

