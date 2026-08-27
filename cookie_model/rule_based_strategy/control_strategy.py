import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import csv
import os
from pathlib import Path
from pandas import DataFrame
import pandas as pd
from scipy.optimize import NonlinearConstraint, Bounds

"""
script to calculate time series according to simple rule based strategies

variables:
t6_oil = return flow temperature to HTHP HE
m_oil =  secondary massflow
t_cooling = const. =  temperature of heat source
n = rot. speed
"""

############################
# input / global variables #
############################
# ToDo: set values identical to GAMS
time_step = 3600 # sec
t_cooling = 75 # celsius
oil_heat_capacity = 2210 # J/(kg K)
storage_efficiency = 0.9
storage_heat_capacity = 1025 # in J/ (kg K)
storage_mass = 600000# mass in kg?
storage_temp_start = 250 # celsius, storage temperate for t = 0

#charging parameters
n_charge = 1.42 # fuer wechselhafte Woche
#bei 8 kg/s am Verdampfer inlet ca. 275°C -> T1 > 300°C -> storage muss kühlen -> massenstromregelung
storage_temp_max = 300 # in celsius, 290 for windschwache Woche
m_oil_charge = 6.0# fuer wechselhafte Woche
dQ_max = 1.9e06 # max value during charge - ToDo no impact
reduce_dQ_for_storage_heat = storage_temp_max * 1 # if storage is hotter than value -> dQ_max * 0.7 - currently not used

#discharging parameters
storage_temp_min = 185 # in celsius
m_oil_discharge = 15# scaled value 1/3, # 13, 8.5 windschwach
t6_temp_diff_factor = 0.9 # control discharge energy
t6_max = 230 #max temp reached when discharging -> set max discharging dQ

#cost evaluation
#mean_costs = 10 # for test array
#mean_costs_charging = 90 # charging costs < 90 -> use this as cost indicator! (24h case)
#mean_costs_charging = 145 # fuer windschwache Woche
#mean_costs_charging = 50 # fuer wechselhafte Woche - nur laden wenn sehr guenstig
mean_costs_charging = 115 # fuer neue Woche

#mean_costs_discharging = -20 # discharging costs > - 20
cost_deviation_tolerance_charging = 0.11#values to set neutral zone without charge/discharge # 0.2, 0.15 windschwach
cost_deviation_tolerance_discharging = 0.32 #discharge dQ/dt > charging dQ/dt -> higher value # 0.3, 0.2 windschwach

# Repository-relative paths make the script portable across Linux and Windows.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DATA_PATH = SCRIPT_DIR.parent / "data_24h_test.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
GAMS_RESULTS_PATH = (
    SCRIPT_DIR
    / "GAMS_results"
    / "Results_Woche_2707_0208_costs_control_strategy_IPOPT_75.xlsx"
)

# Read the 24-hour test data bundled with this repository.
input_data = pd.read_excel(INPUT_DATA_PATH, sheet_name="Tabelle1")
required_columns = {"WKA", "Pe"}
missing_columns = required_columns.difference(input_data.columns)
if missing_columns:
    raise ValueError(
        f"{INPUT_DATA_PATH} is missing required columns: {sorted(missing_columns)}"
    )
windpower_list = input_data["WKA"].to_numpy(dtype=float)
price_list = input_data["Pe"].to_numpy(dtype=float)

# 1) decisions based only on charging costs - considers available wind power

charging_costs = []
discharging_costs = []
i = 0
for windpower in windpower_list:
    charging_costs.append((4962 - windpower)/1000 * price_list[i]) #dQ charging = 4962kW
    discharging_costs.append((1500 - windpower)/1000 * price_list[i]) #dQ discharging = 1500kW
    i = i + 1

# read drectly from excel with pandas
def read_data_from_excel():
    """
    read data from excel files with pandas
    """
    df = pd.read_excel(INPUT_DATA_PATH, sheet_name="Tabelle1")
    DF=df.to_numpy()

    dataSize=len(DF)
    price_list = np.zeros(dataSize)
    windpower_list = np.zeros(dataSize)

    for i in range(dataSize):
        price_list[i]=DF[i][2]
        windpower_list[i]=DF[i][1]
        #if windpower_list[i] > 3500:
        #    windpower_list[i] = 3500

    charging_costs = []
    discharging_costs = []
    i = 0
    for windpower in windpower_list:
        charging_costs.append((4962 - windpower)/1000 * price_list[i]) #dQ charging = 4962kW
        discharging_costs.append((1500 - windpower)/1000 * price_list[i]) #dQ discharging = 1500kW
        i = i + 1

    return price_list, windpower_list, charging_costs, discharging_costs


price_list, windpower_list, charging_costs, discharging_costs = read_data_from_excel()

# manually generated price array (replaces random periodic function or list read from file)
#ToDo: read price array from input data - integrate grid power, costs and emissions for time period
#price_list = [10, 14, 17, 16, 9, 5, 4, 3, 5, 7, 7, 12, 14, 16, 18,  11, 10,  6,  5,  9,  13,  15,  14,  12]


################
# script start #
################

def surrogate_hthp(t6_oil, m_oil, n, t_cooling):
    """
    calculates current T1 and power input for HTHP when storage is charged/discharged
    """
    # 2nd degree surrogate models
    #t1_oil = -38.933755+0.877264*t6_oil-5.331574*m_oil+0.036030*t_cooling+165.449158*n+0.000280*t6_oil**2+0.016184*t6_oil*m_oil-0.000086*t6_oil*t_cooling-0.299898*t6_oil*n+0.469151*m_oil**2-0.004047*m_oil*t_cooling-11.193857*m_oil*n-0.000042*t_cooling**2+0.064724*t_cooling*n+59.403322*n**2
    #power_hthp =  361.476716+0.717079*t6_oil+4.931400*m_oil+0.364875*t_cooling-1346.130947*n+0.002674*t6_oil**2-0.027499*t6_oil*m_oil+0.001967*t6_oil*t_cooling-2.420658*t6_oil*n-0.435295*m_oil**2-0.002693*m_oil*t_cooling+11.155371*m_oil*n+0.000023*t_cooling**2-1.387271*t_cooling*n+1633.790655*n**2

    # 3rd degree surrogate models
    #t1_oil = 77.437428 + 1.107914 * t6_oil - 1.115863 * m_oil + 0.038518 * t_cooling - 246.328572 * n + 0.001554 * t6_oil ** 2 + 0.028503 * t6_oil * m_oil - 0.003749 * t6_oil * t_cooling - 1.048811 * t6_oil * n + 0.921916 * m_oil ** 2 - 0.008469 * m_oil * t_cooling - 26.477567 * m_oil * n + 0.005607 * t_cooling ** 2 + 0.009142 * t_cooling * n + 559.220129 * n ** 2 - 0.000002 * t6_oil ** 3 - 0.000016 * t6_oil ** 2 * m_oil + 0.000006 * t6_oil ** 2 * t_cooling - 0.000146 * t6_oil ** 2 * n - 0.001245 * t6_oil * m_oil ** 2 + 0.000013 * t6_oil * m_oil * t_cooling + 0.016314 * t6_oil * m_oil * n + 0.000005 * t6_oil * t_cooling ** 2 + 0.000134 * t6_oil * t_cooling * n + 0.272784 * t6_oil * n ** 2 - 0.039955 * m_oil ** 3 + 0.000251 * m_oil ** 2 * t_cooling + 0.918848 * m_oil ** 2 * n + 0.000006 * m_oil * t_cooling ** 2 - 0.003948 * m_oil * t_cooling * n - 3.175710 * m_oil * n ** 2 - 0.000024 * t_cooling ** 3 - 0.000773 * t_cooling ** 2 * n + 0.074594 * t_cooling * n ** 2 - 152.884856 * n ** 3
    #power_hthp = 571.716224 + 6.552350 * t6_oil - 1.318223 * m_oil + 1.974182 * t_cooling - 3128.935458 * n - 0.016957 * t6_oil ** 2 - 0.009456 * t6_oil * m_oil + 0.004877 * t6_oil * t_cooling - 5.868292 * t6_oil * n - 0.617052 * m_oil ** 2 - 0.006267 * m_oil * t_cooling + 17.239440 * m_oil * n - 0.005209 * t_cooling ** 2 - 4.156330 * t_cooling * n + 3669.370590 * n ** 2 + 0.000025 * t6_oil ** 3 + 0.000045 * t6_oil ** 2 * m_oil - 0.000005 * t6_oil ** 2 * t_cooling + 0.002762 * t6_oil ** 2 * n + 0.003612 * t6_oil * m_oil ** 2 + 0.000009 * t6_oil * m_oil * t_cooling - 0.097232 * t6_oil * m_oil * n - 0.000007 * t6_oil * t_cooling ** 2 + 0.000318 * t6_oil * t_cooling * n + 1.411628 * t6_oil * n ** 2 + 0.046495 * m_oil ** 3 + 0.000666 * m_oil ** 2 * t_cooling - 1.842242 * m_oil ** 2 * n + 0.000000 * m_oil * t_cooling ** 2 - 0.011183 * m_oil * t_cooling * n + 23.703166 * m_oil * n ** 2 + 0.000026 * t_cooling ** 3 + 0.000400 * t_cooling ** 2 * n + 1.209888 * t_cooling * n ** 2 - 778.996190 * n ** 3

    # 3rd degree surrogate models
    t1_oil = 95.9612 + 0.93433 * t6_oil - 0.327753 * m_oil + 0.0146542 * t_cooling - 271.354 * n + 0.00104853 * t6_oil ** 2 + 0.0211819 * t6_oil * m_oil - 0.706122 * t6_oil * n + 1.04924 * m_oil ** 2 - 0.00388073 * m_oil * t_cooling - 29.4801 * m_oil * n + 0.0595068 * t_cooling * n + 562.428 * n ** 2 - 0.000716825 * t6_oil ** 2 * n - 0.00148575 * t6_oil * m_oil ** 2 + 0.0229386 * t6_oil * m_oil * n + 0.203578 * t6_oil * n ** 2 - 0.0405702 * m_oil ** 3 + 0.881391 * m_oil ** 2 * n - 2.18172 * m_oil * n ** 2 - 151.476 * n ** 3
    power_hthp = 127.87 + 2.06342 * t6_oil + 2.55723 * m_oil + 0.756419 * t_cooling - 1164.84 * n - 0.0168942 * t6_oil * m_oil - 2.60579 * t6_oil * n - 0.540713 * m_oil ** 2 + 13.3204 * m_oil * n - 1.3829 * t_cooling * n + 1556.66 * n ** 2

    return t1_oil, power_hthp

def surrogate_steam_generator(m_oil):
    """
    returns T3 and T4 for given oil massflow
    """
    temp_steam_generator_in = 201.915 + 1819.32 / m_oil
    temp_steam_generator_out = -188.403 / m_oil + 196.3
    return temp_steam_generator_in, temp_steam_generator_out

def calc_x1_bypass(t1, t3, storage_temp, storage_efficiency):
    """
    returns bypass values for charging process to match inlet temp. of steam generator
    """
    t1 = t1 + 273
    t3 = t3 + 273
    storage_temp = storage_temp + 273
    t2 = t1 - storage_efficiency * (t1 - storage_temp)
    #t3 = t1 * x1 + t2 * (1 - x1)  # x1 = 1: 100% bypass
    x1 = (t3/t1 - t2/t1) / (1 - t2/t1)
    return x1

def calc_storage_heat(t_in, storage_temp, massflow, fluid_heat_capacity, storage_heat_capacity, storage_mass,
                      time_step):
    """
    returns new storage temp and transferred heat for timestep
    """
    t_out = t_in - storage_efficiency * (t_in - storage_temp)
    transferred_heat = massflow * fluid_heat_capacity * (t_in - t_out)
    new_storage_temp = storage_temp + transferred_heat / (storage_heat_capacity * storage_mass) * time_step

    return transferred_heat, new_storage_temp

def calc_x2_bypass(t4, t6, storage_temp, storage_efficiency):
    """
    returns bypass values for discharging process to match inlet temp. of HTHP HE
    """
    t4 = t4 + 273
    t6 = t6 + 273
    storage_temp = storage_temp  + 273
    t5 = t4 - storage_efficiency * (t4 - storage_temp)
    #t6 = t4 * x2 + t5 * (1 - x2)  # x2 = 1: 100% bypass
    x2 = (t6 / t4 - t5 / t4) / (1 - t5 / t4)
    return x2

def find_discharge_rotational_speed(n, t1_target, t6_oil, m_oil, t_cooling):
    """
    objective function to iteratively calculate n for desired t6 and t1 when discharging
    """
    #t1_oil = -38.933755+0.877264*t6_oil-5.331574*m_oil+0.036030*t_cooling+165.449158*n+0.000280*t6_oil**2+0.016184*t6_oil*m_oil-0.000086*t6_oil*t_cooling-0.299898*t6_oil*n+0.469151*m_oil**2-0.004047*m_oil*t_cooling-11.193857*m_oil*n-0.000042*t_cooling**2+0.064724*t_cooling*n+59.403322*n**2
    #t1_oil = 77.437428 + 1.107914 * t6_oil - 1.115863 * m_oil + 0.038518 * t_cooling - 246.328572 * n + 0.001554 * t6_oil ** 2 + 0.028503 * t6_oil * m_oil - 0.003749 * t6_oil * t_cooling - 1.048811 * t6_oil * n + 0.921916 * m_oil ** 2 - 0.008469 * m_oil * t_cooling - 26.477567 * m_oil * n + 0.005607 * t_cooling ** 2 + 0.009142 * t_cooling * n + 559.220129 * n ** 2 - 0.000002 * t6_oil ** 3 - 0.000016 * t6_oil ** 2 * m_oil + 0.000006 * t6_oil ** 2 * t_cooling - 0.000146 * t6_oil ** 2 * n - 0.001245 * t6_oil * m_oil ** 2 + 0.000013 * t6_oil * m_oil * t_cooling + 0.016314 * t6_oil * m_oil * n + 0.000005 * t6_oil * t_cooling ** 2 + 0.000134 * t6_oil * t_cooling * n + 0.272784 * t6_oil * n ** 2 - 0.039955 * m_oil ** 3 + 0.000251 * m_oil ** 2 * t_cooling + 0.918848 * m_oil ** 2 * n + 0.000006 * m_oil * t_cooling ** 2 - 0.003948 * m_oil * t_cooling * n - 3.175710 * m_oil * n ** 2 - 0.000024 * t_cooling ** 3 - 0.000773 * t_cooling ** 2 * n + 0.074594 * t_cooling * n ** 2 - 152.884856 * n ** 3
    t1_oil = 95.9612+0.93433*t6_oil-0.327753*m_oil+0.0146542*t_cooling-271.354*n+0.00104853*t6_oil**2+0.0211819*t6_oil*m_oil-0.706122*t6_oil*n+1.04924*m_oil**2-0.00388073*m_oil*t_cooling-29.4801*m_oil*n+0.0595068*t_cooling*n+562.428*n**2-0.000716825*t6_oil**2*n-0.00148575*t6_oil*m_oil**2+0.0229386*t6_oil*m_oil*n+0.203578*t6_oil*n**2-0.0405702*m_oil**3+0.881391*m_oil**2*n-2.18172*m_oil*n**2-151.476*n**3
    return np.abs(t1_oil - t1_target)

def objective_function_charging(variables, storage_temp, wind_power):
    """
    objective function to set multiple parameters to compensate high wind speeds
    """
    m_oil = variables[0] * 16
    n = variables[1] * 1.53
    #t_source = variables[2] * 100
    t_source = t_cooling
    # calc. steam generator inlet and outlet
    t3, t4 = surrogate_steam_generator(m_oil * 3)
    t6 = t4
    t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_source)
    # calc. bypass
    x1 = calc_x1_bypass(t1, t3, storage_temp, storage_efficiency)
    # reduce oil massflow to further heat up storage until upper limit is reached
    x2 = 0
    m_oil_storage = (1 - x1) * m_oil
    # calc time step / storage heat and updated storage temp
    dQ, updated_storage_temp = calc_storage_heat(t1, storage_temp, m_oil_storage * 3, oil_heat_capacity,
                                                 storage_heat_capacity, storage_mass,
                                                 time_step)
    charging_dQ_max = dQ_max*1e-06
    charging_dQ = dQ * 1e-06
    #return 1/dQ
    #return abs(1 / dQ)
    #return (1/dQ) * (-1)
    return abs(1 / charging_dQ)
    #return abs(charging_dQ_max - dQ)
    #return 1/power_hthp/1000

def constr_dQ(variables, storage_temp, wind_power):
    """
    objective function to set multiple parameters to compensate high wind speeds
    """
    m_oil = variables[0] * 16
    n = variables[1] * 1.53
    #t_source = variables[2] * 100
    t_source = t_cooling
    # calc. steam generator inlet and outlet
    t3, t4 = surrogate_steam_generator(m_oil * 3)
    t6 = t4
    t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_source)
    # calc. bypass
    x1 = calc_x1_bypass(t1, t3, storage_temp, storage_efficiency)
    # reduce oil massflow to further heat up storage until upper limit is reached
    x2 = 0
    m_oil_storage = (1 - x1) * m_oil
    # calc time step / storage heat and updated storage temp
    dQ, updated_storage_temp = calc_storage_heat(t1, storage_temp, m_oil_storage * 3, oil_heat_capacity,
                                                 storage_heat_capacity, storage_mass,
                                                 time_step)
    charging_dQ_max = dQ_max
    if storage_temp > reduce_dQ_for_storage_heat:
        charging_dQ_max = 0.7 * charging_dQ_max

    print ("dQ cosntr " + str((dQ - charging_dQ_max)*1e-06))
    return (dQ - charging_dQ_max) * 1e-06

def constr_x1_1(variables, storage_temp, wind_power):
    """
    objective function to set multiple parameters to compensate high wind speeds
    """
    m_oil = variables[0] * 16
    n = variables[1] * 1.53
    #t_source = variables[2] * 100
    t_source = t_cooling
    # calc. steam generator inlet and outlet
    t3, t4 = surrogate_steam_generator(m_oil * 3)
    t6 = t4
    t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_source)
    # calc. bypass
    x1 = calc_x1_bypass(t1, t3, storage_temp, storage_efficiency)
    # reduce oil massflow to further heat up storage until upper limit is reached
    x2 = 0
    m_oil_storage = (1 - x1) * m_oil
    # calc time step / storage heat and updated storage temp
    dQ, updated_storage_temp = calc_storage_heat(t1, storage_temp, m_oil_storage * 3, oil_heat_capacity,
                                                 storage_heat_capacity, storage_mass,
                                                 time_step)
    return x1 + 0.00001 - 1

def constr_x1_0(variables, storage_temp, wind_power):
    """
    objective function to set multiple parameters to compensate high wind speeds
    """
    m_oil = variables[0] * 16
    n = variables[1] * 1.53
    #t_source = variables[2] * 100
    t_source = t_cooling
    # calc. steam generator inlet and outlet
    t3, t4 = surrogate_steam_generator(m_oil * 3)
    t6 = t4
    t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_source)
    # calc. bypass
    x1 = calc_x1_bypass(t1, t3, storage_temp, storage_efficiency)
    # reduce oil massflow to further heat up storage until upper limit is reached
    x2 = 0
    m_oil_storage = (1 - x1) * m_oil
    # calc time step / storage heat and updated storage temp
    dQ, updated_storage_temp = calc_storage_heat(t1, storage_temp, m_oil_storage * 3, oil_heat_capacity,
                                                 storage_heat_capacity, storage_mass,
                                                 time_step)
    return (x1 + 0.00001) *  (-1)

def constr_costs(variables, storage_temp, wind_power):
    """
    objective function to set multiple parameters to compensate high wind speeds
    """
    m_oil = variables[0] * 16
    n = variables[1] * 1.53
    #t_source = variables[2] * 100
    t_source = t_cooling
    # calc. steam generator inlet and outlet
    t3, t4 = surrogate_steam_generator(m_oil * 3)
    t6 = t4
    t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_source)
    # calc. bypass
    x1 = calc_x1_bypass(t1, t3, storage_temp, storage_efficiency)
    # reduce oil massflow to further heat up storage until upper limit is reached
    x2 = 0
    m_oil_storage = (1 - x1) * m_oil
    # calc time step / storage heat and updated storage temp
    dQ, updated_storage_temp = calc_storage_heat(t1, storage_temp, m_oil_storage * 3, oil_heat_capacity,
                                                 storage_heat_capacity, storage_mass,
                                                 time_step)
    diff = wind_power - 3 * power_hthp
    print(diff)
    return wind_power - 3 * power_hthp

def objective_function_wind_compensation(variables, storage_temp, wind_power):
    """
    objective function to set multiple parameters to compensate high wind speeds
    """
    m_oil = variables[0] * 16
    n = variables[1] * 1.53
    #t_source = variables[2] * 100
    t_source = t_cooling
    # calc. steam generator inlet and outlet
    t3, t4 = surrogate_steam_generator(m_oil * 3)
    t6 = t4
    t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_source)
    # calc. bypass
    x1 = calc_x1_bypass(t1, t3, storage_temp, storage_efficiency)
    # reduce oil massflow to further heat up storage until upper limit is reached
    x2 = 0
    m_oil_storage = (1 - x1) * m_oil
    # calc time step / storage heat and updated storage temp
    dQ, updated_storage_temp = calc_storage_heat(t1, storage_temp, m_oil_storage * 3, oil_heat_capacity,
                                                 storage_heat_capacity, storage_mass,
                                                 time_step)
    dQ_charge = dQ
    #print (str(m_oil) + ", " +str(n) + ", " +str(t_source))
    diff = (wind_power - 3 * power_hthp) / 1000
    print("\t" + str(diff))
    #return abs(wind_power - 3 * power_hthp)*1e-03
    return diff**2
    #return abs(diff)



def objective_function_discharging(variables):
    """
    objective function to set multiple parameters to compensate high wind speeds
    """

    return 0


def simple_control_strategy(price, storage_temp, wind_power):
    """
    if prices are low and storage is not fully charged -> charge
    if prices are high and storage is not fully discharged -> discharge
    else -> HTHP in design mode, storage disconnected
    """
    # charge # ToDo: Bedingung ...
    if (price < mean_costs_charging  * ( 1 - cost_deviation_tolerance_charging) and storage_temp < storage_temp_max *0.95):

        #variables = [1,1]

        # bounds BP_Flow, air flow temp, flow split for T100 coupling
        # obj-function eg. electric_output or max. HP-Flow (ToDo)
        lower_bounds = [0.3125, 0.8]
        upper_bounds = [0.8, 1]
        bnds = Bounds(lower_bounds, upper_bounds, keep_feasible=False)

        x0 = [0.53, 0.97]
        x0 = [m_oil_charge/16, n_charge/1.53]
        #if storage_temp > reduce_dQ_for_storage_heat:
        #    x0 = [0.35, 0.95, 0.6]

        #variables = np.array([x0])

        # constraint definitions
        #nlc_max = NonlinearConstraint(constr_dQ, 0, charging_dQ_max, keep_feasible=False)
        #nlc_x1 = NonlinearConstraint(constr_x1, 0, 1, keep_feasible=False)
        #nlc_costs = NonlinearConstraint(constr_costs, 0, np.inf, keep_feasible=False)

        cons_x1_1 = ({'type': 'ineq', 'fun': constr_x1_1, 'args': (storage_temp, wind_power), 'keep_feasible' : True})
        cons_x1_0 = ({'type': 'ineq', 'fun': constr_x1_0, 'args': (storage_temp, wind_power), 'keep_feasible' : True})
        cons_costs = ({'type': 'ineq', 'fun': constr_costs, 'args': (storage_temp, wind_power), 'keep_feasible' : True})
        cons_max = ({'type': 'ineq', 'fun': constr_dQ, 'args': (storage_temp, wind_power), 'keep_feasible' : True})

        #cons = (cons_x1_0, cons_x1_1, cons_costs, cons_max)

        #res = optimize.minimize(objective_function_charging, x0,  constraints = cons, args=(storage_temp, wind_power),
        # 		               method='trust-constr', jac=None, bounds = bnds, options = {'verbose': 1, 'initial_constr_penalty': 2.0, 'initial_tr_radius' : 0.5})

        res = optimize.minimize(objective_function_charging, x0, constraints=(cons_x1_1, cons_x1_0, cons_max), args=(storage_temp, wind_power),
                                   method='SLSQP', jac=None, bounds=bnds)
        # options={'disp': None,  'eps': 1e-01}
        print(res)

        #print(res)
        # recalculate
        variables = res.x
        n = variables[1] * 1.53
        m_oil = variables[0] * 16
        #t_source = variables[2] * 100
        t_source = t_cooling
        # calc. steam generator inlet and outlet
        t3, t4 = surrogate_steam_generator(m_oil * 3)
        t6 = t4
        t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_source)
        # calc. bypass
        x1 = calc_x1_bypass(t1, t3, storage_temp, storage_efficiency)
        # reduce oil massflow to further heat up storage until upper limit is reached
        x2 = 0
        m_oil_storage = (1 - x1) * m_oil
        # calc time step / storage heat and updated storage temp
        dQ, updated_storage_temp = calc_storage_heat(t1, storage_temp, m_oil_storage * 3, oil_heat_capacity,
                                                     storage_heat_capacity, storage_mass,
                                                     time_step)
        print("optimization results charging: ")
        print("\tx1: " + str(x1) + ", storage massflow: " + str(m_oil_storage)+ ", T_source: " + str(t_source))
        print("\tHTHP power: " + str(power_hthp) + ", T1: " + str(t1))
        print("\tdQ: " + str(dQ) + ", updated storage temp: " + str(updated_storage_temp))

    elif (wind_power > 3 * 965): #and storage_temp < storage_temp_max): # charge to prevent neg. costs for high wind speeds

        # ToDo: negative x1 Werte ???
        lower_bounds = [0.3125, 0.8]
        upper_bounds = [0.7, 1]
        bnds = Bounds(lower_bounds, upper_bounds, keep_feasible=False)

        #bnds = ((0.3125, 1),(0.8, 1), (0.6, 1))

        x0 = [0.35, 0.93]
        #if storage_temp > 280:
        #    x0 = [0.35, 0.95, 0.6]
        #x0 = [0.35, 0.95, 0.6]
        variables = np.array([x0])

        # constraint definitions
        #nlc_x1 = NonlinearConstraint(constr_x1, 0, 1, keep_feasible=False)
        #nlc_max = NonlinearConstraint(constr_dQ, 0, dQ_max, keep_feasible=False)

        cons_x1_1 = ({'type': 'ineq', 'fun': constr_x1_1, 'args': (storage_temp, wind_power), 'keep_feasible' : True})
        cons_x1_0 = ({'type': 'ineq', 'fun': constr_x1_0, 'args': (storage_temp, wind_power),'keep_feasible' : True})
        cons_costs = ({'type': 'ineq', 'fun': constr_costs, 'args': (storage_temp, wind_power),'keep_feasible' : True}) # ToDo or equality constraint
        cons_max = ({'type': 'ineq', 'fun': constr_dQ, 'args': (storage_temp, wind_power),'keep_feasible' : True})

        #res = optimize.minimize(objective_function_wind_compensation, x0, constraints=(cons_x1_1, cons_x1_0, cons_max), args=(storage_temp, wind_power),
        #                           method='trust-constr', jac=None, bounds=bnds, options = {'verbose': 1, 'initial_constr_penalty': 1.0, 'initial_tr_radius' : 1.0})
        res = optimize.minimize(objective_function_wind_compensation, x0, constraints=(cons_x1_1, cons_x1_0), args=(storage_temp, wind_power),
                                   method='SLSQP', jac=None, bounds=bnds, options={'disp': None, 'maxiter': 1000}, tol = 1e-04)

        print(res)
        # recalculate
        variables = res.x
        n = variables[1] * 1.53
        m_oil = variables[0] * 16
        #t_source = variables[2] * 100
        t_source = t_cooling
        # calc. steam generator inlet and outlet
        t3, t4 = surrogate_steam_generator(m_oil * 3)
        t6 = t4
        t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_source)
        # calc. bypass
        x1 = calc_x1_bypass(t1, t3, storage_temp, storage_efficiency)
        # reduce oil massflow to further heat up storage until upper limit is reached
        x2 = 0
        m_oil_storage = (1 - x1) * m_oil
        # calc time step / storage heat and updated storage temp
        dQ, updated_storage_temp = calc_storage_heat(t1, storage_temp, m_oil_storage * 3, oil_heat_capacity,
                                                         storage_heat_capacity, storage_mass,
                                                         time_step)
        print("optimization for wind compensation: ")
        print("\tx1: " + str(x1) + ", storage massflow: " + str(m_oil_storage)+ ", T_source: " + str(t_source))
        print("\tHTHP power: " + str(power_hthp) + ", T1: " + str(t1))
        print("\tdQ: " + str(dQ) + ", updated storage temp: " + str(updated_storage_temp))

    # discharge ToDo: general approach, T6 estimation, t6 limitation ....
    elif(price > mean_costs_charging * ( 1 + cost_deviation_tolerance_discharging) and storage_temp > storage_temp_min and wind_power < 3 * 970):
        t_source = t_cooling
        m_oil = m_oil_discharge  # some value
        print("discharge initiated with estimation of T6:")
        # calc. steam generator inlet and outlet
        t3, t4 = surrogate_steam_generator(m_oil * 3)
        t6 = t6_temp_diff_factor * (storage_temp - t4) + t4
        # t6  = storage_temp - 10
        if t6 > t6_max:
            t6 = t6_max
        diff = - 1 #variable to initiate loop to prevent grid feed <-> lower t6 until costs are positive
        while diff < 0:
            t1_target = t3
            # find rotational speed for given return flow temp. and oil-massflow
            n = optimize.minimize_scalar(find_discharge_rotational_speed, method = 'bounded', bounds = (0.7, 1.6),  args=(t1_target, t6, m_oil,
                    t_cooling))
            n = n.x
            t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_cooling)
            t6 = t6 - 0.02
            if t6 < t4 - 0.2:
                break
            diff = 3 * power_hthp - wind_power
        print("\tT6 set to: " + str(t6) + ", t3 set to: " + str(t1_target))
        print("\tHTHP power: " + str(power_hthp) + " with n set to: " + str(n) + ", T1: " + str(t1))
        # calc. bypass
        x1 = 0
        x2 = calc_x2_bypass(t4, t6, storage_temp, storage_efficiency)
        m_oil_storage = (1 - x2) * m_oil
        print("\tx2: " + str(x2) + ", storage massflow: " + str(m_oil_storage))
        # calc time step / storage heat and updated storage temp
        dQ, updated_storage_temp = calc_storage_heat(t4, storage_temp, m_oil_storage * 3, oil_heat_capacity, storage_heat_capacity,
                                                     storage_mass, time_step)
        print("\tdQ: " + str(dQ) + ", updated storage temp: " + str(updated_storage_temp))

    else:
        # disconnect storage - hthp in design point
        t_source = t_cooling
        print("storage disconnected - hthp in design op")
        m_oil = 6.3
        t3, t4 = surrogate_steam_generator(m_oil * 3)
        t6 = t4
        n = 1.28
        t1, power_hthp = surrogate_hthp(t6, m_oil, n, t_cooling)
        print("\tHTHP power: " + str(power_hthp) + ", T1: " + str(t1))
        updated_storage_temp = storage_temp
        dQ = 0
        x1 = x2 = 0

    return (
        updated_storage_temp,
        dQ,
        power_hthp * 3,
        x1,
        x2,
        n,
        m_oil * 3,
        t_source,
        t1,
        t3,
        t4,
        t6,
    )


def perform_loop(charging_costs_list): # decisions in control strategy based on charging costs
    """
    loop for current price input and update of time dependent storage temperature
    append results to lists
    """
    storage_temp_list = []
    dQ_list = []
    power_hthp_list = []
    n_list = []
    m_oil_list = []
    t_source_list = []
    t1_list = []
    t3_list = []
    t4_list = []
    t6_list = []
    next_storage_temp_list = []

    x1_list = []
    x2_list = []


    storage_temp = storage_temp_start
    i = 0
    costs = [] # cost balance for each time step

    for price in charging_costs_list:
        print("time step: " + str(i + 1))
        storage_temp_list.append(storage_temp)
        #new_storage_temp, dQ, power_hthp, x1, x2, n, m_oil = simple_control_strategy(price_list[i], storage_temp)
        (
            new_storage_temp,
            dQ,
            power_hthp,
            x1,
            x2,
            n,
            m_oil,
            t_source,
            t1,
            t3,
            t4,
            t6,
        ) = simple_control_strategy(price, storage_temp, windpower_list[i])
        costs.append((power_hthp - windpower_list[i])/1000 * price_list[i])
        storage_temp = new_storage_temp
        dQ_list.append(dQ)
        t_source_list.append(t_source)
        power_hthp_list.append(power_hthp)
        n_list.append(n)
        m_oil_list.append(m_oil)
        x1_list.append(x1)
        x2_list.append(x2)
        t1_list.append(t1)
        t3_list.append(t3)
        t4_list.append(t4)
        t6_list.append(t6)
        next_storage_temp_list.append(new_storage_temp)
        i = i + 1

    #print("\nresults: ")
    print(dQ_list)
    print(storage_temp_list)
    #print(power_hthp_list)

    return (
        dQ_list,
        storage_temp_list,
        next_storage_temp_list,
        power_hthp_list,
        x1_list,
        x2_list,
        n_list,
        m_oil_list,
        costs,
        t_source_list,
        t1_list,
        t3_list,
        t4_list,
        t6_list,
    )

#####################
#perform calculation#
#####################

"""
ToDo: rename t_cooling to t_source 
TODO: how to set t6 when discharging
ToDo: storage_start_temperature = storage_temp(t=n)
ToDo: Read input data (wind and grid price) -> cost indicator -> calc. needed grid power and integrate costs
ToDo: more complex control strategy -> esp. for charging process
    - variable oil massflow for target dQ
"""


# test input data
#storage_temp, dQ, power_hthp = simple_control_strategy(10, 270)


def summarize_rl_constraint_violations(
    storage_temp_before,
    storage_temp_after,
    dQ,
    power_hthp,
    wind_power,
    x1,
    x2,
    n,
    m_oil_times_three,
    t1,
    t3,
    t4,
    t6,
):
    """Calculate the rule-based trajectory's violations in RL-model units."""
    storage_temp_before = np.asarray(storage_temp_before, dtype=float)
    storage_temp_after = np.asarray(storage_temp_after, dtype=float)
    dQ = np.asarray(dQ, dtype=float)
    power_hthp = np.asarray(power_hthp, dtype=float)
    wind_power = np.asarray(wind_power, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n = np.asarray(n, dtype=float)
    m_oil = np.asarray(m_oil_times_three, dtype=float) / 3.0
    t1 = np.asarray(t1, dtype=float)
    t3 = np.asarray(t3, dtype=float)
    t4 = np.asarray(t4, dtype=float)
    t6 = np.asarray(t6, dtype=float)

    q_ch = np.maximum(dQ, 0.0) / 1000.0
    q_dch = np.maximum(-dQ, 0.0) / 1000.0
    t2 = t1 - storage_efficiency * (t1 - storage_temp_before)
    t5 = t4 - storage_efficiency * (t4 - storage_temp_before)
    f4_residual = t3 - (t2 * (1.0 - x1) + t1 * x1)
    f5_residual = t6 - (t4 * x2 + t5 * (1.0 - x2))
    grid_power = power_hthp - wind_power

    violations = {
        "bypass_F10": np.maximum(1.0 - (x1 + x2), 0.0).sum(),
        "complementarity_F11": np.maximum(q_ch * q_dch - 1.0e-6, 0.0).sum(),
        "flow_bounds_F6_F7_kW": (
            np.maximum(q_ch - 5000.0, 0.0) + np.maximum(q_dch - 5000.0, 0.0)
        ).sum(),
        "inlet_temperature_F5_C": (
            np.maximum(177.0 - t6, 0.0) + np.maximum(t6 - 250.0, 0.0)
        ).sum(),
        "storage_temperature_F14_C": (
            np.maximum(183.0 - storage_temp_after, 0.0)
            + np.maximum(storage_temp_after - 324.0, 0.0)
        ).sum(),
        "grid_power_F3_kW": (
            np.maximum(-grid_power, 0.0) + np.maximum(grid_power - 5000.0, 0.0)
        ).sum(),
        "F4_residual_C": np.abs(f4_residual).sum(),
        "F5_residual_C": np.abs(f5_residual).sum(),
        "compressor_speed_C16": (
            np.maximum(0.8 - n, 0.0) + np.maximum(n - 1.53, 0.0)
        ).sum(),
        "mass_flow_C15": (
            np.maximum(5.0 - m_oil, 0.0) + np.maximum(m_oil - 16.0, 0.0)
        ).sum(),
        "terminal_storage_F14_C": abs(storage_temp_after[-1] - storage_temp_start),
    }
    normalized_total = (
        violations["bypass_F10"]
        + violations["complementarity_F11"] / 5000.0**2
        + violations["flow_bounds_F6_F7_kW"] / 5000.0
        + violations["inlet_temperature_F5_C"] / (250.0 - 177.0)
        + violations["storage_temperature_F14_C"] / (324.0 - 183.0)
        + violations["grid_power_F3_kW"] / 5000.0
        + violations["F4_residual_C"] / (324.0 - 183.0)
        + violations["F5_residual_C"] / (324.0 - 183.0)
        + violations["compressor_speed_C16"] / (1.53 - 0.8)
        + violations["mass_flow_C15"] / (16.0 - 5.0)
        + violations["terminal_storage_F14_C"] / (324.0 - 183.0)
    )
    violations["normalized_total"] = normalized_total
    return violations


# perform loop over time period and integrate costs
(
    dQ_list,
    storage_temp_list,
    next_storage_temp_list,
    power_hthp_list,
    x1,
    x2,
    n_list,
    m_oil_list,
    costs,
    t_source,
    t1_list,
    t3_list,
    t4_list,
    t6_list,
) = perform_loop(charging_costs)
overall_costs = sum(costs)
print ("overall operational costs: " + str(overall_costs))
constraint_violations = summarize_rl_constraint_violations(
    storage_temp_list,
    next_storage_temp_list,
    dQ_list,
    power_hthp_list,
    windpower_list,
    x1,
    x2,
    n_list,
    m_oil_list,
    t1_list,
    t3_list,
    t4_list,
    t6_list,
)
constraint_report = "RL-equivalent constraint violations (24-hour total)\n" + "\n".join(
    f"{name}={value:.6f}" for name, value in constraint_violations.items()
) + "\n"
(OUTPUT_DIR / "operational_cost.txt").write_text(
    f"total_operational_cost={overall_costs:.6f}\n\n{constraint_report}",
    encoding="utf-8",
)
(OUTPUT_DIR / "constraint_violations.txt").write_text(
    constraint_report,
    encoding="utf-8",
)

#####################
#    plot results   #
#####################

# read results from excel with pandas
def read_gams_results_from_excel():
    """
    read data from excel files with pandas
    """
    if not GAMS_RESULTS_PATH.exists():
        print(f"GAMS comparison file not found; skipping comparison plots: {GAMS_RESULTS_PATH}")
        return None

    df = pd.read_excel(GAMS_RESULTS_PATH, sheet_name="Tabelle1", header=3)



    DF=df.to_numpy()

    dataSize=len(DF)
    gams_storage_temp = np.zeros(dataSize)
    gams_Qch = np.zeros(dataSize)
    gams_Qdch = np.zeros(dataSize)
    gams_n = np.zeros(dataSize)
    gams_m_oil = np.zeros(dataSize)

    for i in range(dataSize):
        gams_storage_temp[i]=DF[i][4]
        gams_Qch[i] = DF[i][13]
        gams_Qdch[i] = DF[i][16]
        gams_n[i] = DF[i][28]
        gams_m_oil[i] = DF[i][31] * 3

    return gams_storage_temp, gams_Qch, gams_Qdch, gams_n, gams_m_oil


gams_results = read_gams_results_from_excel()
if gams_results is not None:
    gams_storage_temp, gams_Qch, gams_Qdch, gams_n, gams_m_oil = gams_results

plt.figure(1, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
plt.bar(x_pos, price_list, zorder = 3, ec = 'black', label = 'grid power price')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()

plt.savefig(OUTPUT_DIR / "price_input.png")

plt.figure(2, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
plt.bar(x_pos, dQ_list, zorder = 3, color = "darkorange",ec = 'black', label = 'transferred heat')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "dQ.png")

plt.figure(3, dpi=200, figsize=(15,3))
#plt.grid(b=False, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes.xaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
#plt.bar(x_pos, storage_temp_list, zorder = 3, color = "darkgreen",ec = 'black', label = 'storage temperature')
plt.plot(x_pos, storage_temp_list, '-', zorder = 3, color = "darkorange", linewidth=3, label = 'control strategy')
if gams_results is not None and len(gams_storage_temp) == len(storage_temp_list):
    plt.plot(x_pos, gams_storage_temp, '-', zorder = 3, color = "black", label = 'gams')
#plt.xticks(x_pos, hours, rotation=0)
plt.ylim(bottom = 180, top = 340)
plt.xlim(left = 0, right = max(len(hours) - 1, 1))
plt.xticks(np.arange(0, len(hours), max(len(hours) // 6, 1)))
# Custom the subplot layout
plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right = 0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "storage_temp.png")
plt.savefig(OUTPUT_DIR / "storage_temp.pdf")

plt.figure(4, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
plt.bar(x_pos, power_hthp_list, zorder = 3, color = "black",ec = 'black', label = 'hthp power')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "hthp_power.png")

plt.figure(5, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
plt.bar(x_pos, x1, zorder = 3, color = "lightgrey",ec = 'black', label = 'x1')
plt.bar(x_pos, x2, zorder = 3, color = "dimgray", ec = 'black', label = 'x2')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "x1x2.png")

plt.figure(6, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
#plt.bar(x_pos, n_list, zorder = 3, color = "orchid",ec = 'black', label = 'rotational speed')
plt.plot(x_pos, n_list, '-', zorder = 3, linewidth = 3, color = "orchid", label = 'rotational speed')
if gams_results is not None and len(gams_n) == len(n_list):
    plt.plot(x_pos, gams_n, '--', zorder = 3, color = "orchid", label = 'gams')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.ylim(bottom = 0.8)
plt.legend()
plt.savefig(OUTPUT_DIR / "rot_speed.png")

plt.figure(7, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
#plt.bar(x_pos, m_oil_list, zorder = 3, color = "deepskyblue",ec = 'black', label = 'oil massflow')
plt.plot(x_pos, m_oil_list, '-', zorder = 3, color = "deepskyblue", linewidth = 3, label = 'oil massflow')
if gams_results is not None and len(gams_m_oil) == len(m_oil_list):
    plt.plot(x_pos, gams_m_oil, '--', zorder = 3, color = "deepskyblue", label = 'gams')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
plt.ylim(bottom = 10)
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "m_oil.png")

plt.figure(8, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
plt.bar(x_pos, charging_costs, zorder = 3, align='center', color = "darkred",ec = 'black', label = 'charging costs')
plt.bar(x_pos, discharging_costs, zorder = 5, align='center', color = "deepskyblue",ec = 'black', label = 'discharging costs')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
#plt.ylim(bottom = 10)
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "costs.png")

plt.figure(9, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
plt.bar(x_pos, windpower_list, zorder = 3, color = "lightgreen",ec = 'black', label = 'windpower')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
#plt.ylim(bottom = 10)
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "windpower.png")

plt.figure(10, dpi=200, figsize=(12,5))
#plt.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
plt.bar(x_pos, costs, zorder = 3, color = "gold",ec = 'black', label = 'operational costs')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
#plt.ylim(bottom = 10)
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "operational_costs.png")

plt.figure(11, dpi=200, figsize=(12,5))
#plt.grid(b=False, linestyle='--', linewidth=0.5, zorder = 0)
axes = plt.gca()
axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
hours = list(range(1,len(price_list) + 1))
x_pos = np.arange(len(hours))
#plt.bar(x_pos, t_source, zorder = 3, color = "red",ec = 'black', label = 'T source')
plt.plot(x_pos, t_source, '-', zorder = 3, color = "red", label = 'T source')
#plt.xticks(x_pos, hours, rotation=0)
# Custom the subplot layout
plt.ylim(bottom = 60)
plt.subplots_adjust(bottom=0.1, top=0.95)
plt.legend()
plt.savefig(OUTPUT_DIR / "Tsource.png")

if gams_results is not None and len(gams_Qch) == len(price_list):
    plt.figure(12, dpi=200, figsize=(12,5))
    axes = plt.gca()
    axes.yaxis.grid(visible=True, linestyle='--', linewidth=0.5, zorder = 0)
    hours = list(range(1,len(price_list) + 1))
    x_pos = np.arange(len(hours))
    plt.bar(x_pos, gams_Qch, zorder = 3, color = "darkorange",ec = 'black', label = 'gams: transferred heat')
    plt.bar(x_pos, -gams_Qdch, zorder = 3, color = "darkorange",ec = 'black')
    plt.subplots_adjust(bottom=0.1, top=0.95)
    plt.legend()
    plt.savefig(OUTPUT_DIR / "dW_gams.png")

plt.show()






