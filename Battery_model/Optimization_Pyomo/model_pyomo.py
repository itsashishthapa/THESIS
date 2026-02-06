# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:16:08 2025

@author: Loukas Kyriakidis

Optimization of the battery model in Pyomo

"""

import time

import matplotlib.pyplot as plt
import pandas as pd
from pyomo.environ import *

# Read the txt file (assuming space or tab delimited)
df = pd.read_csv("input_data/electricity_price.txt", sep='\\s+', header=None)

# get electricity price data
prices = df[1].tolist()[72:96]  
T = len(prices)

model = ConcreteModel()

# sets
model.T = RangeSet(0, T-1)
price_dict = {t: prices[t]/(1e6) for t in range(T)}

# Parameters
P_max = 20000 # in W
eta = 0.9  # charge/discharge efficiency
E_max = 2e5*3600 # in Ws
dt = 3600 # in s
SOC_0 = 0.5 # initial SOC
k=500 # paramter for smoothing

# variables
model.u = Var(model.T, bounds=(-1, 1))
model.SOC = Var(model.T, bounds=(0, 1))
model.P = Var(model.T, bounds=(-20000, 20000))
model.P_actual = Var(model.T, bounds=(-20000, 20000))

# Objective: Minimize cost from grid
def objective(m):
    return sum(price_dict[t] * m.P[t] for t in m.T)

# Constraint: P = u * P_max
def power_eq(m, t):
    return m.P[t] == P_max*(0.5*(1+tanh(k*(m.u[t]-0.0025)))*m.u[t]*0.5*(1+tanh(k*(0.995-m.SOC[t])))+0.5*(1-tanh(k*(m.u[t]+0.0025)))*m.u[t]*0.5*(1+tanh(k*(m.SOC[t]-0.005))))
model.power_constraint = Constraint(model.T, rule=power_eq)

# Constraint for P_actual 
def p_actual(m, t):
    return m.P_actual[t] == 0.9 * m.P[t] 
model.p_actual_constraint = Constraint(model.T, rule=p_actual)

# SOC dynamics
def soc_dynamics(m, t):
    if t == 0:
        return m.SOC[t] == SOC_0 + dt * m.P_actual[t]/E_max
    else:
        return m.SOC[t] == m.SOC[t-1] + dt * m.P_actual[t]/E_max
model.soc_constraint = Constraint(model.T, rule=soc_dynamics)

# Solve with IPOPT
model.obj = Objective(rule=objective, sense=minimize)
solver = SolverFactory('ipopt')
solver.options['max_iter']= 1000 # number of iterations of the local solver 
start_time = time.time() # starting time for optimization
res = solver.solve(model, tee=True)
end_time = time.time() # end time for optimization
elapsed_time = end_time - start_time # total time of optimization
obj = value(model.obj)

print("Objective value:", round(obj, 2), "Euro")
print(f"Optimization time: {elapsed_time:.2f} seconds")

"""
u = []
SOC = []
P = []
P_actual=[]

for i in range(len(prices)):
   u.append(value(model.u[i]))
   SOC.append(value(model.SOC[i]))
   P.append(value(model.P[i]))
   P_actual.append(value(model.P_actual[i]))

plt.figure(figsize=(10,6))
plt.plot(u, label="Exact piecewise", linewidth=2)
plt.grid(True)
plt.savefig('figure.png')
"""
