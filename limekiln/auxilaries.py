import numpy as np
import math
import pandas as pd
from scipy import interpolate, integrate
import os
import pickle
import importlib.resources


## General constants
Rgas = 8314.463  # universal gas constant [J/(kmol*K)]
sigma_sb = 5.67e-8  # Stefan-Boltzmann constant [W/m2/K4]

p0 = 1.01325e5  # Normal pressure [Pa]
T0 = 273.15     # Normal Temperature [K]

T_amb = 273.15 + 20  # ambient temperature [K]
p_amb = 1.01325e5  # ambient pressure [Pa]

## Constant component properties
barM_g = np.array([28, 32, 44, 18, 16])  # molar mass gaseous components for N2, O2, CO2, H2O, CH4 [kg/kmol]
barM_s = np.array([100, 56])  # molar mass solid components for CaCO3, CaO [kg/kmol]
# rho_LS = 2700  # density limestone [kg/m3], actual value not needed
Dp = 1.77e-4  # pore diffusion coefficient of CO2 in lime [m2/s]


## Reaction data
deltaHr = 168e6  # enthalpy of reaction [J/kmol]
k = 0.90e-2  # reaction coefficient m/s
nu_calc = np.array([0, 0, 1, 0, 0])  # stiochiometric coefficients calcination
nu_comb = np.array([0, -2, 1, 2, -1])  # stoichiometric coefficients combustion

## Functions to compute properties of gaseous components
def get_pure_thermal_conductivity(p, T):
    """
    Calculates thermal conductivity of gaseous components
    """
    lambda0 = np.array([0.024, 0.025, 0.017, 0.016, 0.030])  # thermal conductivity at reference temperature [W/m/K]
    nlambda = np.array([0.76, 0.80, 1.04, 1.42, 1.37])  # exponent
    return (T.reshape(-1, 1) / T0) ** nlambda * lambda0  # temperature dependency

def get_pure_dynamic_viscisity(p, T):
    """
    Calculates dynamic viscosity of gaseous components
    """
    mu0 = np.array([16.8, 19.7, 14.4, 8.7, 10.4]) * 1e-6  # dynamic viscosity at reference temperature [kg/m/s]
    nmu = np.array([0.67, 0.67, 0.77, 1.13, 0.77])  # exponent
    return (T.reshape(-1, 1) / T0) ** nmu * mu0  # temperature dependency

def get_pure_density(p, T):
    """
    Calculates density of gaseous components
    """
    rho0 = np.array([1.26, 1.44, 1.98, 0.81, 0.71])  # density at reference temperature
    return np.reshape(p, (-1, 1)) / p0 * T0 / np.reshape(T, (-1, 1))  * rho0  # pressure and temperature dependency


def get_diffusion_coefficient(p, T):
    """
    Calculates diffusion coefficient of CO2 in air
    """
    DCO20 = 0.14e-4  # diffusion coefficient at reference temperature [m2/s]
    nd = 1.71  # exponent
    return (T / T0) ** (nd + 1) * DCO20  # temperature dependency

## Functions to compute properties of gas mixtures
def get_thermal_conductivity(p, T, yi):
    """
    Calculates thermal conductivity of gas mixture depending on molar composition
    """
    return np.sum(get_pure_thermal_conductivity(p, T) * yi, axis=1)

def get_dynamic_viscosity(p, T, yi):
    """
    Calculates dynamic viscosity of gas mixture depending on molar composition
    """
    mui = get_pure_dynamic_viscisity(p, T)
    return np.sum(mui * yi * np.sqrt(barM_g), axis=-1) / np.sum(yi * np.sqrt(barM_g), axis=-1)

def get_density(p, T, yi):
    """
    Calculates density of gas mixture depending on molar composition
    """
    return np.sum(get_pure_density(p, T) * yi, axis=1)

def get_specific_heat_capacity(p, T, yi):
    """
    Calculates specific heat capacity of gas mixture depending on molar composition
    """
    rho_i = get_pure_density(p, T)
    return np.sum(get_specific_heat_capacity_gases(T) * yi * rho_i, axis=1) / np.sum(rho_i * yi, axis=1)

## Functions to compute dimensionless numbers
def get_superficial_velocity(p, T, N_Gi):
    """
    Calculates superficial velocity
    """
    yi = N_Gi / N_Gi.sum(axis=-1, keepdims=True)
    rho_g = get_density(p, T, yi)
    return np.sum(barM_g * N_Gi, axis=-1) / (rho_g * A_k)

def get_Schmidt(p, T, yi):
    """
    Calculates Schmidt number
    """
    mu = get_dynamic_viscosity(p, T, yi)
    rho = get_density(p, T, yi)
    D = get_diffusion_coefficient(p, T)
    return mu / rho / D

def get_Reynolds(p, T, N_gi):
    """
    Calculates Reynolds number
    """
    u = get_superficial_velocity(p, T, N_gi)
    yi = N_gi / N_gi.sum(axis=-1, keepdims=True)
    mu = get_dynamic_viscosity(p, T, yi)
    rho = get_density(p, T, yi)
    return u * d_p * rho / (mu * phi)

def get_Prandtl(p, T, yi):
    """
    Calculates Reynolds number
    """
    mu = get_dynamic_viscosity(p, T, yi)
    c_p = get_specific_heat_capacity(p, T, yi)
    lambda_ = get_thermal_conductivity(p, T, yi)
    return mu * c_p / lambda_

def Nusselt_bed_Jeschar(Re, Pr):
    """"
    Calculates Nusselt number in bed according to Jeschar
    """
    Nu = 2 + 1.12 * (Re ** 0.5) * (Pr ** 0.33) * ((1 - phi) / phi) ** 0.5 + 0.0058 * Re * Pr ** 0.4
    # Nu = 2 + 1.12 * (Re ** 0.5) * (Pr ** 0.33) * ((1 - phi) / phi) ** 0.5 + 0.005 * Re
    return Nu

def Nusselt_bed_VDI(Re, Pr):
    """"
    Calculates Nusselt number in bed according to VDI
    """
    Nu_turb = (0.037 * Re**0.8 * Pr) / (1 + 2.443 * Re**(-0.1) * (Pr**(2/3) - 1))
    Nu_lam = 0.664 * Re ** 0.5 + Pr**(1/3)
    Nu_ek = 2 + (Nu_lam**2 + Nu_turb**2)**0.5
    f_a = 1 + 1.5 * (1 - phi)
    Nu = Nu_ek * f_a
    return Nu

def Nusselt_bed(p, T, N_gi):
    """"
    Wrapper function for calculation of Nusselt number in bed
    """
    Re = get_Reynolds(p, T, N_gi)
    yi = N_gi / N_gi.sum(axis=-1, keepdims=True)
    Pr = get_Prandtl(p, T, yi)
    # return Nusselt_bed_Jeschar(Re, Pr)
    return Nusselt_bed_VDI(Re, Pr)

def Sherwood_bed(p, T, N_gi):
    """"
    Wrapper function for calculation of Sherwood number in bed
    """
    Re = get_Reynolds(p, T, N_gi)
    yi = N_gi / N_gi.sum(axis=-1, keepdims=True)
    Sc = get_Schmidt(p, T, yi)
    # due to analogy of heat and mass transfer we can use the Nusselt correlation
    # with the Schmidt number replacing the Prandtl number
    return Nusselt_bed_VDI(Re, Sc)
    # return Nusselt_bed_Jeschar(Re, Sc)

## A few more auxilary functions
def get_pressure_drop(p, T, N_Gi):
    def ergun(p, T, N_Gi):
        v_s =  get_superficial_velocity(p, T, N_Gi)
        yi = N_Gi / N_Gi.sum(axis=-1, keepdims=True)
        rho = get_density(p, T, yi)
        mu = get_dynamic_viscosity(p, T, yi)
        return 150 * (1 - phi)**2 / phi**3 * mu * v_s / d_p**2 + 1.75 * (1 - phi) / phi**3 * rho * v_s**2 / d_p

    def molerus(p, T, N_Gi):
        r0_delta = (0.95 / (1 - phi)**(1/3) - 1) ** (-1)
        Re = get_Reynolds(p, T, N_Gi)
        Eu = 24 / Re * (1 + 0.692 * (r0_delta + 0.5 * r0_delta**2)) + \
             4 / Re**0.5 * (1 + 0.12 * r0_delta**1.5) + \
             0.4 + 0.891 * r0_delta * Re**(-0.1)
        yi = N_Gi / N_Gi.sum(axis=-1, keepdims=True)
        rho = get_density(p, T, yi)
        v_s = get_superficial_velocity(p, T, N_Gi)
        return 0.75 * Eu * (1 - phi) / phi**2 * rho * v_s**2 / d_p
    
    # return np.zeros(np.shape(p))  # no pressure drop
    return ergun(p, T, N_Gi)
    # return molerus(p, T, N_Gi)

def get_Xf(x):
    """"
    Calculates fuel conversion and its derivative
    """
    z = np.copy(x)
    if isinstance(z, float):
        z = np.array([z])

    # flame parameters
    b = 2
    a = -4 * math.log(10)

    z[z < L_p] = L_p
    X_f = 1 - np.exp(a * ((z - L_p) / L_r)**b)
    dX_f = - a * b / L_r * np.power((z - L_p) / L_r, b - 1) * np.exp(a * ((z - L_p) / L_r)**b)
    return X_f, dX_f

def p_eq(Tsf):
    """
    Calculates equilibirum partial pressure of CO2
    """
    # Tsf[Tsf < 1] = 1
    p0 = 3.03e12  # Stoßkonstante für Gleichgewichtsdruckberechnung Pa
    if isinstance(Tsf, (int, float)):
        peq = p0 * math.exp(-deltaHr / Rgas / Tsf)
    else:
        peq = p0 * np.exp(-deltaHr / Rgas / Tsf)
    return peq

## Functions for computation of specific heat capacities and enthalpies of components and mixture
with importlib.resources.open_text(__package__, 'ShomateParameters.csv') as fh:
    shomate = pd.read_csv(fh, header=0, index_col=0)
shomate = shomate.drop(['G'])  # for consistency to other implemented functions

def get_Shomate_parameters(T):
    """
    Get Shomate parameters [A, B, C, D, E, F, H] at a specific temperature based on the CSV file
    """
    cols = 'N2 O2 CO2 H2O CH4'.split(' ')
    idx = shomate.index
    c_df = pd.DataFrame(np.full((len(idx), len(cols)), np.nan), columns=cols, index=idx)
    for comp in cols:
        tmp = shomate.loc[:, [col for col in shomate if col.startswith(comp)]]
        T_min = tmp.loc['Tmin', :].min()
        T_max = tmp.loc['Tmax', :].max()
        # if temprature in range
        if T_min <= T <= T_max:
            for col in tmp:
                if tmp.at['Tmin', col] <= T <= tmp.at['Tmax', col]:
                    c_df.loc[:, comp] = tmp.loc[:, col]
                    break
        # if temperature below range
        elif T < T_min:
            col = tmp.loc['Tmin', :].idxmin()
            c_df.loc[:, comp] = tmp.loc[:, col]
        # if temperature above range
        elif T > T_max:
            col = tmp.loc['Tmax', :].idxmax()
            c_df.loc[:, comp] = tmp.loc[:, col]
    return c_df.drop(['Tmin', 'Tmax']).to_numpy()

def get_pure_specific_enthalpy(T):
    """
    Calculates specific enthalpy of gaseous components
    """
    [A, B, C, D, E, F, H] = get_Shomate_parameters(T)
    t = T / 1000
    barH_0 = np.array([0, 0, -393.52e6, -241.83e6, -74.87e6])  # [J/kmol]
    barH = (A*t + B/2*t**2 + C/3*t**3 + D/4*t**4 - E/t + F - H) *1e6 + barH_0  # [J/kmol]
    return barH / barM_g  # [J/kg]

def get_pure_specific_heat_capacities(T):
    """
    Calculates specific heat capacities of gaseous components
    """
    [A, B, C, D, E, _, _] = get_Shomate_parameters(T)
    t = T / 1000
    return (A + B * t + C * t ** 2 + D * t ** 3 + E / (t ** 2)) / (barM_g * 1e-3)  # [J/kg/K]

def get_specific_heat_capacity_CaO(T):
    """
    Calculates specific heat capacity of CaO
    """
    t = T / 1000
    A, B, C, D, E = (49.95403, 4.887916, -0.352056, 0.046187, -0.825097)
    return (A + B * t + C * t ** 2 + D * t ** 3 + E / t ** 2) / 0.056  # [J/kg/K]

def get_specific_enthalpy_CaO(T):
    """
    Calculates specific enthalpy of CaO
    """
    t = T / 1000
    A, B, C, D, E, F, H = (49.95403, 4.887916, -0.352056, 0.046187, -0.825097, -652.9718, -635.0894)
    barH_0 = -635.09e6  # [J/kmol]
    barM = 56  # [kg/kmol]
    barH = (A*t + B/2*t**2 + C/3*t**3 + D/4*t**4 - E/t + F - H) * 1e6 + barH_0  # [J/kmol]
    return barH / barM  # [J/kg]

def get_specific_heat_capacity_CaCO3(T):
    """
    Calculates specific heat capacity of CaCO3 according to Jacobs
    """
    A, B, C, D, E = (-184.79, 0.32322, -3.6882e6, -1.2974e-4, 3883.5)
    barM = 100  # [kg/kmol]

    if T <= 775:
        barC_p = (A + B*T + C/T**2 + D*T**2 + E/T**0.5) * 1e3  # [J/kmol/K]
    else:
        dT = T - 775
        T = 775
        barC_p775 = (A + B*T + C/T**2 + D*T**2 + E/T**0.5) * 1e3  # [J/kmol/K]
        dbarC_pdT775 = (B - 2*C/T**3 + 2*D*T - 0.5*E/T**1.5) * 1e3  # slope at T = 775 K [J/kmol/K2]
        barC_p = barC_p775 + dbarC_pdT775 / 2* dT  # [J/kmol/K]
    return barC_p / barM  # [J/kg/K]

def get_specific_enthalpy_CaCO3(T):
    """
    Calculates specific enthalpy of CaCO3 according to Jacobs
    """
    barH_0 = -1207e6
    barM = 100  # [kg/kmol]
    return integrate.quad(get_specific_heat_capacity_CaCO3, 298.15, T)[0] + barH_0 / barM

## Interpolate heat capacities and enthalpies for speedup
T = np.arange(start=270, stop=2501, step=1)
h_gas = np.full((len(T), 5), np.nan)
cp_gas = np.full((len(T), 5), np.nan)
h_solid = np.full((len(T), 2), np.nan)
cp_solid = np.full((len(T), 2), np.nan)
for qty in ['h_gas', 'cp_gas', 'cp_solid', 'h_solid']:
    file = f'.\pickle\{qty}.pkl'
    if os.path.exists(file):
        with open(file, 'rb') as fh:
            exec(f'{qty} = pickle.load(fh)')
    else:
        for i in range(len(T)):
            if qty == 'h_gas':
                h_gas[i, :] = get_pure_specific_enthalpy(T[i])
            elif qty == 'cp_gas':
                cp_gas[i, :] = get_pure_specific_heat_capacities(T[i])
            elif qty == 'h_solid':
                h_solid[i, :] = [get_specific_enthalpy_CaCO3(T[i]), get_specific_enthalpy_CaO(T[i])]
            elif qty == 'cp_solid':
                cp_solid[i, :] = [get_specific_heat_capacity_CaCO3(T[i]), get_specific_heat_capacity_CaO(T[i])]
        with open(file, 'wb') as fh:
            exec(f'pickle.dump({qty}, fh)')

get_specific_enthalpy_gases = interpolate.interp1d(T, h_gas, axis=0, fill_value="extrapolate")
get_specific_heat_capacity_gases = interpolate.interp1d(T, cp_gas, axis=0, fill_value="extrapolate")
get_specific_enthalpy_solids = interpolate.interp1d(T, h_solid, axis=0, fill_value="extrapolate")
get_specific_heat_capacity_solids = interpolate.interp1d(T, cp_solid, axis=0, fill_value="extrapolate")

def get_heat_of_combustion(T):
    h_T = get_specific_enthalpy_gases(T)
    barH_T = h_T * barM_g
    heat_of_combustion = - np.sum(barH_T * nu_comb) / barM_g[-1]
    return heat_of_combustion

## Decide for simulations parameters of either Hallak 2019 or Krause 2017
hallak = True  # get data for Hallak 2019
krause = False  # get data for Krause et al 2017
assert(hallak ^ krause)  # ^: exclusive or

## Kiln dimensions
if hallak:
    r1 = 2  # inner kiln diameter [m]
    A_k = math.pi * r1 ** 2  # cross-section area [m2]

    L_p = 5  # length of preheating zone [m]
    L_r = 6  # length of reaction zone [m]
    L_c = 5  # length of cooling zone [m]
if krause:
    A_k = 1.6 * 0.6 * 8  # rectangle of 1.6 m * 0.6 m is approx 1/8 of cross section
    r1 = math.sqrt(A_k / math.pi)

    L_p = 2.2 + 3.6
    L_r = 6
    L_c = 6.2
s_ref = 0.2  # thickness anti-wear layer [m]
s_ins = 0.064  # thickness insulation layer [m]
r2 = r1 + s_ref
r3 = r2 + s_ins
lambda_ref = 1.5  # thermal conductivity anti-wear layer [m]
lambda_ins = 0.7  # thermal conductivity insulation layer [m]

alpha_amb = 10  # convective heat transfer coefficient at outer kiln wall [W/m2/K]
egrad = 0.8  # emissivity [-]

## Properties of particles and bed
if hallak:
    phi = 0.43  # void fraction of bed [-]
    d_p = 80e-3  # particle diameter [m]
if krause:
    phi = 0.35
    d_p_min = 50e-3
    d_p_max = 90e-3
    d_pi = np.linspace(d_p_min, d_p_max, 9)
    # particle sizes are evenly mass distributed
    # since density is the same for all particles, mass fraction equations volume fraction
    # from that, we can determine the ratio of number of all particles
    V_i_single = math.pi / 6 * d_pi**3
    A_i_single = math.pi * d_pi**2
    # ratio particles in class to total number or particles
    n_i = (1 / V_i_single) / np.sum(1 / V_i_single)
    V_i = n_i * V_i_single
    A_i = n_i * A_i_single
    d_p = 6 * V_i.sum() / A_i.sum()
r_p = d_p / 2  # particle radius [m]
s = 6 / d_p  # volume-specific surface area [m2/m3]
lambda_ls = 0.76  # thermal conductivity [W/m2/K]

## Feed streams
y_O2 = 0.2096  # molar fraction oxygen in air
y_CO2 = 0.0004  # molar fraction carbon dioxide in air
y_N2 = 1 - y_O2 - y_CO2  # molar fraction nitrogen in air
y_air = np.array([y_N2, y_O2, y_CO2, 0, 0])  # N2, O2, CO2, H2O, CH4
barM_air = np.sum(y_air * barM_g)  # molar mass of air [kg/kmol]
w_air = y_air * barM_g / barM_air  # mass fractions in air

if hallak:
    dotM_ls_in = 1014e3 / (24 * 3600) / 2  # kg/s, per shaft (1014 t/d)
    # dotM_CH4 = p0 / (Rgas * T0) * (2206 / 3600) * 16  # kg/s (2206 Nm3/h)
    dotM_CH4 = 22.91e6 / get_heat_of_combustion(273.15 + 25)  # heat power 22.91 MW
    dotM_airCombustion = p0 / (Rgas * T0) * (26300 / 3600) * barM_air # kg/s (26300 Nm3/h)
    dotM_airCool_in = p0 / (Rgas * T0) * (8100 / 3600) * barM_air  # kg/s, per shaft (8100 Nm3/h)
if krause:
    dotM_ls_in = 8 * 89e3 / (24 * 3600) / 2  # kg/s, per shaft (8 * 89 t/d)
    # dotM_CH4 = p0 / (Rgas * T0) * (8 * 216 / 3600) * 16  # kg/s (8 * 216 Nm3/h)
    dotM_CH4 = 8 * 2022e3 / get_heat_of_combustion(273.15 + 25)  # heat power 8 * 2022kW
    dotM_airCombustion = p0 / (Rgas * T0) * (8 * 2155 / 3600) * barM_air # kg/s (8 * 2155 Nm3/h)
    dotM_airCool_in = p0 / (Rgas * T0) * (8 * 1368 / 3600) * barM_air / 2  # kg/s, per shaft (8 * 1368 Nm3/h)