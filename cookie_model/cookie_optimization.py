# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:21:03 2023

"""
 
# Use case for ECOS2023 paper: 24h, multistart, IPOPT
# Model problem from original paper (electrified steam generation)
# HTHP-based (surrogate-based model) Keksfabrik combined with sensible thermal storage (physical model) and evaporator (surrogate-based model)
# thermal storage - without binary variable (charge/discharge via complementarity condition)
# time step: dt=3600 --> 1h
# data (price, wind power) import from "WKA_Pe_merged.csv" file

from pyomo.environ import *
import random
import sys
import numpy as np
import math
from pathlib import Path
from pandas import DataFrame
import pandas as pd
import timeit

# read input data
###################################################
input_file = Path(__file__).with_name('WKA_Pe_merged.csv')
df = pd.read_csv(input_file, nrows=24)

dataSize=len(df)
power_price = df['Pe'].to_numpy(dtype=float)
power_WKA = df['WKA'].to_numpy(dtype=float)

# loop over k random-based start initialization
k=10

Result = np.zeros((dataSize*17,k))
Objective_values = np.zeros(k)
Solver_times = np.zeros(k)
counter=0
start = timeit.default_timer() # starts the timer
while counter < k:
 
    model = ConcreteModel()

    # definition: parameter
    ###################################################
    model.Time = RangeSet(0, dataSize-1)
    EPS=1.e-6
    effCh=0.9
    effDch=0.9
    S0=250
    Ms=600000
    cp=1.025
    cf=2.21     
    deltaT=3600
    
    
    # definition: input data
    ###################################################
    powerprice = dict(enumerate(power_price))
    powerWKA = dict(enumerate(power_WKA))
    
    model.Pe = Param(model.Time, initialize=powerprice)
    model.WKA = Param(model.Time, initialize=powerWKA)

    
    # definition: decision variables
    ###################################################
    model.Thx1  = Var(model.Time,domain=NonNegativeReals,bounds = (177,250),initialize=random.uniform(177,250))
    model.Thx2  = Var(model.Time,domain=NonNegativeReals,bounds = (75,75),initialize=random.uniform(75,75))
    model.M1  = Var(model.Time,domain=NonNegativeReals,bounds = (5,16),initialize=random.uniform(5,16))
    model.P  = Var(model.Time,domain=NonNegativeReals,bounds = (0,5000),initialize=random.uniform(0,5000))
    model.N  = Var(model.Time,domain=NonNegativeReals,bounds = (0.8,1.53),initialize=random.uniform(0.8,1.53))
#    model.S  = Var(model.Time,domain=NonNegativeReals,bounds = (183,400),initialize=random.uniform(183,400))
    model.x1  = Var(model.Time,domain=NonNegativeReals,bounds = (0,1),initialize=random.uniform(0,1))
    model.x2  = Var(model.Time,domain=NonNegativeReals,bounds = (0,1),initialize=random.uniform(0,1))
    model.Qch  = Var(model.Time,domain=NonNegativeReals,bounds = (0,5000),initialize=random.uniform(0,5000))
    model.Qdch  = Var(model.Time,domain=NonNegativeReals,bounds = (0,5000),initialize=random.uniform(0,5000))
    model.T1Out  = Var(model.Time,domain=NonNegativeReals,bounds = (239,400),initialize=random.uniform(239,400))
    model.T2Out  = Var(model.Time,domain=Reals,bounds = (-60,90),initialize=random.uniform(-60,90))
#    model.T2  = Var(model.Time,domain=NonNegativeReals,bounds = (183,400),initialize=random.uniform(183,400))
#    model.T5  = Var(model.Time,domain=NonNegativeReals,bounds = (183,400),initialize=random.uniform(183,400))
#    model.T0  = Var(model.Time,domain=NonNegativeReals,bounds = (183,400),initialize=random.uniform(183,400))
    model.T3  = Var(model.Time,domain=NonNegativeReals,bounds = (239,324),initialize=random.uniform(239,324))
    model.T4  = Var(model.Time,domain=NonNegativeReals,bounds = (183,193),initialize=random.uniform(183,193))
    
    # modified boundary conditions
    model.S  = Var(model.Time,domain=NonNegativeReals,bounds = (183,324),initialize=random.uniform(183,324))
    model.T2  = Var(model.Time,domain=NonNegativeReals,bounds = (183,332),initialize=random.uniform(183,332))
    model.T5  = Var(model.Time,domain=NonNegativeReals,bounds = (183,332),initialize=random.uniform(183,332))
    model.T0  = Var(model.Time,domain=NonNegativeReals,bounds = (183,324),initialize=random.uniform(183,324))
    
    # definiton: objective --> minimize operating costs
    ###################################################
    def objective(m):
        return sum((model.Pe[t]/1000*model.P[t]) for t in model.Time) 
    
    
    # definiton: boundary conditions
    ###################################################
    
    
    # HTHP: hot side
    def equality_cons_F1_rule(m,t):
        return model.T1Out[t] == 95.9612+0.93433*model.Thx1[t]-0.327753*model.M1[t]+0.0146542*model.Thx2[t]-271.354*model.N[t]+0.00104853*model.Thx1[t]**2+0.0211819*model.Thx1[t]*model.M1[t]-0.706122*model.Thx1[t]*model.N[t]+1.04924*model.M1[t]**2-0.00388073*model.M1[t]*model.Thx2[t]-29.4801*model.M1[t]*model.N[t]+0.0595068*model.Thx2[t]*model.N[t]+562.428*model.N[t]**2-0.000716825*model.Thx1[t]**2*model.N[t]-0.00148575*model.Thx1[t]*model.M1[t]**2+0.0229386*model.Thx1[t]*model.M1[t]*model.N[t]+0.203578*model.Thx1[t]*model.N[t]**2-0.0405702*model.M1[t]**3+0.881391*model.M1[t]**2*model.N[t]-2.18172*model.M1[t]*model.N[t]**2-151.476*model.N[t]**3
    model.equality_cons_F1 = Constraint(model.Time, rule=equality_cons_F1_rule)
    
    # HTHP: cold side
    def equality_cons_F2_rule(m,t):
        return model.T2Out[t] == 93.3958-0.00692483*model.Thx1[t]-0.770173*model.M1[t]+1.30277*model.Thx2[t]-183.866*model.N[t]+0.00313225*model.Thx1[t]*model.M1[t]+0.234082*model.Thx1[t]*model.N[t]+0.106964*model.M1[t]**2-2.34999*model.M1[t]*model.N[t]-0.555879*model.Thx2[t]*model.N[t]+30.2955*model.N[t]**2
    model.equality_cons_F2 = Constraint(model.Time, rule=equality_cons_F2_rule)
    
    # HTHP: electricity consumed
    def equality_cons_F3_rule(m,t):
        return model.P[t]+model.WKA[t] == 3*(127.87+2.06342*model.Thx1[t]+2.55723*model.M1[t]+0.756419*model.Thx2[t]-1164.84*model.N[t]-0.0168942*model.Thx1[t]*model.M1[t]-2.60579*model.Thx1[t]*model.N[t]-0.540713*model.M1[t]**2+13.3204*model.M1[t]*model.N[t]-1.3829*model.Thx2[t]*model.N[t]+1556.66*model.N[t]**2)
    model.equality_cons_F3 = Constraint(model.Time, rule=equality_cons_F3_rule)
    
    # Bypass: HTHP to steam generator
    def equality_cons_F4_rule(m,t):
        return model.T3[t] == model.T2[t]*(1-model.x1[t])+model.T1Out[t]*model.x1[t]
    model.equality_cons_F4 = Constraint(model.Time, rule=equality_cons_F4_rule)
    
    # Bypass: steam generator to HTHP
    def equality_cons_F5_rule(m,t):
        return model.Thx1[t] == model.T4[t]*model.x2[t]+model.T5[t]*(1-model.x2[t])
    model.equality_cons_F5 = Constraint(model.Time, rule=equality_cons_F5_rule)
    
    # storage: charging
    def equality_cons_F6_rule(m,t):
        return model.Qch[t] == (model.T1Out[t]-model.T2[t])*model.M1[t]*3*cf*(1-model.x1[t])
    model.equality_cons_F6 = Constraint(model.Time, rule=equality_cons_F6_rule)
    
    # storage: discharging
    def equality_cons_F7_rule(m,t):
        return model.Qdch[t] == (model.T5[t]-model.T4[t])*model.M1[t]*3*cf*(1-model.x2[t])
    model.equality_cons_F7 = Constraint(model.Time, rule=equality_cons_F7_rule)
    
    # storage: effectivity model charging
    def equality_cons_F8_rule(m,t):
        return model.T2[t] == model.T1Out[t]-effCh*(model.T1Out[t]-model.T0[t])
    model.equality_cons_F8 = Constraint(model.Time, rule=equality_cons_F8_rule)
    
    # storage: effectivity model discharging
    def equality_cons_F9_rule(m,t):
        return model.T5[t] == model.T4[t]-effDch*(model.T4[t]-model.T0[t])
    model.equality_cons_F9 = Constraint(model.Time, rule=equality_cons_F9_rule)
    
    # "help" equation: Bypass-Sum 
    def inequality_cons_F10_rule(m,t):
        return model.x1[t]+model.x2[t] >= 1
    model.inequality_cons_F10 = Constraint(model.Time, rule=inequality_cons_F10_rule)
    
    # "help" equation: complementarity condition --> charging and discharging at the same time is not allowed 
    def inequality_cons_F11_rule(m,t):
        return model.Qch[t]*model.Qdch[t] <= EPS
    model.inequality_cons_F11 = Constraint(model.Time, rule=inequality_cons_F11_rule)
    
    # Steam generator: outlet
    def equality_cons_F12_rule(m,t):
        return model.T4[t] == -188.403/(model.M1[t]*3) + 196.3
    model.equality_cons_F12 = Constraint(model.Time, rule=equality_cons_F12_rule)
    
    # Steam generator: inlet
    def equality_cons_F13_rule(m,t):
        return model.T3[t] == 201.915 + 1819.32/(model.M1[t]*3)
    model.equality_cons_F13 = Constraint(model.Time, rule=equality_cons_F13_rule)
    
    # storage: effectivity model temperature time step before
    def equality_cons_F14_rule(m,t):
        if t==model.Time.first():  
         return model.T0[t] == S0 
        else:
         return model.T0[t] == model.S[t-1]
    model.equality_cons_F14 = Constraint(model.Time, rule=equality_cons_F14_rule)
    
    # storage: storage start == storage end
    def end_storage_rule(m):
        yield model.S[dataSize-1] == S0
    model.end_storage = ConstraintList(rule=end_storage_rule)
    
    # storage: explicit Euler
    def storage_rule(m,t):
      if t==model.Time.first():  
         return model.S[t] == S0+deltaT*model.Qch[t]/(Ms*cp)-deltaT*model.Qdch[t]/(Ms*cp)
      else:
         return model.S[t] == model.S[t-1]+deltaT*model.Qch[t]/(Ms*cp)-deltaT*model.Qdch[t]/(Ms*cp)
    model.storage = Constraint(model.Time, rule=storage_rule) 
    
    # call: solver (ipopt)
    ###################################################
    model.obj = Objective(rule=objective, sense=minimize)
    solver = SolverFactory('ipopt')
    solver.options['max_iter']= 1000 #number of iterations you wish
    res = solver.solve(model, tee=True)
    
    # print: objective value
    ###################################################
    print()
    print('*** Solution *** :')
    print('function value:', value(model.obj))
    
    
    # Result_vector: variables assembled in a row with respect to discrete time point
    ###################################################
    Result_vector= []
    
    for i in range(dataSize):
       Result_vector.append(value(model.P[i]))
       Result_vector.append(value(model.Thx1[i]))
       Result_vector.append(value(model.Thx2[i]))
       Result_vector.append(value(model.M1[i]))
       Result_vector.append(value(model.N[i]))       
       Result_vector.append(value(model.S[i]))
       Result_vector.append(value(model.Qch[i]))
       Result_vector.append(value(model.Qdch[i]))
       Result_vector.append(value(model.T1Out[i]))      
       Result_vector.append(value(model.T2Out[i]))
       Result_vector.append(value(model.T2[i]))
       Result_vector.append(value(model.T5[i]))
       Result_vector.append(value(model.T0[i]))
       Result_vector.append(value(model.x1[i]))
       Result_vector.append(value(model.x2[i]))   
       Result_vector.append(value(model.T3[i]))
       Result_vector.append(value(model.T4[i]))
     
    # convert Result_vector into matrix "Result"
    # save objective value and solver time
    ###################################################
    Result[:,counter] = np.array(Result_vector)
    if (res.solver.status == SolverStatus.ok) and (res.solver.termination_condition == TerminationCondition.optimal):
        Objective_values[counter] = value(model.obj)
        Solver_times[counter] = res.solver.time
    else: 
        Objective_values[counter] = math.inf
        Solver_times[counter] = math.inf
    counter += 1
    
stop = timeit.default_timer() # stops the timer
print('Time in s: ', stop - start)

df = DataFrame({'Solver times': Solver_times, 'Objective values': Objective_values})
df.to_excel('Results_Paper_hybrid_keksfabrik_24h_multistarts_ipopt_new.xlsx', sheet_name='sheet1', index=False)
