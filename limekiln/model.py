import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy import optimize
from scipy.interpolate import *
from .auxilaries import *
import pandas as pd
import math
import sys


def calculateKiln(dotM_ls_in, dotM_CH4_in, dotM_airCombustion_in, dotM_airCool_in):
    """Berechnung der Umsatz- und Temperaturprofile im GGR-Kalkofen.
    Es werden das Umsatzprofil und die Temperaturprofile der Gasphase, der Feststoffphase, der Abgasphase und der Kühl-
    luft bestimmt. Die in dieser Funktion vorkommenden Parameter werden in der Datei "main_vectorized.py" definiert. Es
    wird über eine Schätztemperatur des Übergangskanals zuerst der Vorwärm- und Reaktionsabschnitt berechnet, an-
    schließend der Abkühlabschnitt. Über eine punktuelle Energiebilanz am Übergangskanal wird eine theoretische Misch-
    temperatur des Abgases berechnet. Es wird eine relative Abweichung berechnet zur geschätzten Temperatur. Übersteigt
    diese die festgelegte Toleranz von 1e-4, so wird die neu berechnete Temperatur als Startwert verwendet. Dieser
    Prozess wiederholt sich, bis die Abweichung <1e-4 beträgt. Es werden die Lösungen der Abschnitte ausgegeben.
    """
    results = dict()  # dictionary to be filled with any variables of interest
    ## operating conditions
    # combustion air and natural gas
    M_Gi0 = w_air * dotM_airCombustion_in + np.array([0, 0, 0, 0, 1]) * dotM_CH4_in
    N_cgi0 = M_Gi0 / barM_g  # kmol/s
    N_CH40 = N_cgi0[4]  # kmol/s

    # cooling air
    M_Ai0 = w_air * dotM_airCool_in  # kg/s
    N_Ai0 = M_Ai0 / barM_g  # kmol/s

    # coarse grid for preheating and reaction zone
    z0_p = np.linspace(0, L_p, 51, endpoint=False)
    z0_r = L_p + np.linspace(0, L_r, 101)
    z0 = np.concatenate((z0_p, z0_r))
    # fine grid for preheating and reaction zone
    z0_p = np.linspace(0, L_p, 201, endpoint=False)
    z0_r = L_p + np.linspace(0, L_r, 850)
    z0_fine = np.concatenate((z0_p, z0_r))

    # grid for cooling zone
    z0_cool = z0_fine[-1] + np.linspace(0, L_c, 51)

    # define a few variables here, that are calculated later, so they are accessible
    lnr2_r1 = math.log(r2 / r1)  # log is slow, so precompute
    lnr3_r2 = math.log(r3 / r2)

    # a few solver settings
    max_nodes = z0_fine.size + 25  # prevent excessive refinement, as z0_fine already has a high resolution

    def errorFun_kiln(T_g_mix):
        history = errorFun_kiln.history  # for convencience

        def fun(z, y):  # function for part 1
            # for convenience and readability
            X = y[0, :].copy()
            X[X > 1] = 1  # restrict values
            X[X < 0] = 0
            T_cg = y[1, :]
            T_s = y[2, :]
            T_fg = y[3, :]
            p_cg = y[4, :]
            p_fg = y[5, :]

            # Determine flows, fractions, etc. of combustion gas
            Xf, dXf = get_Xf(z)
            N_cgi = N_cgi0 + np.outer(Xf, nu_comb) * N_CH40 + np.outer(X, nu_calc) * dotM_ls_in / 100  # kmol/s
            M_Gi = N_cgi * barM_g
            x_cg = N_cgi / N_cgi.sum(axis=-1, keepdims=True)
            results['x_cg'] = x_cg
            Nu = Nusselt_bed(p_cg, T_cg, N_cgi)
            alpha_cg = Nu * get_thermal_conductivity(p_cg, T_cg, x_cg) / d_p
            alpha_cg_s = 1 / (1 / alpha_cg + d_p / (2 * 5 * lambda_ls))
            cp_cg_Tcg = get_specific_heat_capacity_gases(T_cg)
            h_cg_Tcg = get_specific_enthalpy_gases(T_cg)
            h_cg_Ts = get_specific_enthalpy_gases(T_s)

            # Determine flows, fractions, etc. of flue gas
            N_fg_i = N_cgi[-1] + 2 * N_Ai0 + np.outer(X[-1] - X, nu_calc) * dotM_ls_in / 100
            M_fg = N_fg_i * barM_g  # kg/s
            x_fg = N_fg_i / N_fg_i.sum(axis=-1, keepdims=True)
            Nu = Nusselt_bed(p_fg, T_fg, N_fg_i)
            alpha_fg = Nu * get_thermal_conductivity(p_fg, T_fg, x_fg) / d_p
            alpha_fg_s = 1 / (1 / alpha_fg + d_p / (2 * 5 * lambda_ls))
            cp_fg_Tfg = get_specific_heat_capacity_gases(T_fg)
            h_fg_Tfg = get_specific_enthalpy_gases(T_fg)

            # Determine flows, fractions, etc. of solid phase
            M_s = np.full((z.size, 2), np.nan)
            M_s[:, 0] = dotM_ls_in * (1 - X)
            M_s[:, 1] = dotM_ls_in * X / 100 * 56
            cp_s_Ts = get_specific_heat_capacity_solids(T_s)

            # Determine dX/dz
            # burning shaft determines calcination process
            Sh = Sherwood_bed(p_cg, T_cg, N_cgi)
            beta = Sh * get_diffusion_coefficient(p_cg, T_cg) / d_p
            P_CO2 = p_cg * x_cg[:, 2]
            f_X = (1 - X) ** (1 / 3)
            c = deltaHr / Rgas * (r_p / lambda_ls * f_X * (1 -f_X) + f_X**2 / alpha_cg) / \
                (r_p / Dp * f_X * (1 - f_X) + f_X**2 / beta + 1 / k)
            T_f = np.full(z.shape, np.nan)
            p = - T_cg - c * P_CO2 / T_cg  # coefficient in p-q-like quadradic formula
            for i in range(z.size):
                def eq(T_fi):
                    q = c[i] * p_eq(T_fi)
                    return T_fi**2 + p[i] * T_fi + q
                
                T_f[i] = optimize.brentq(eq, T_amb-1, max(T_cg[i], T_fg[i])+100)
            dX_dz = (1 - phi) * A_k / dotM_ls_in * 100 / r_p / Rgas * (p_eq(T_f) / T_f - P_CO2 / T_cg) \
                    * 3 * f_X**2 / (r_p / Dp * f_X * (1 - f_X) + f_X**2 / beta + 1 / k)
            
            # Determine outer kiln wall temperature
            def equations2(T_kwi, i):
                # length-specific heat flow combustion gas to outer kiln wall
                q_cg_ko = 1 / (1 / (alpha_cg[i] * r1) + lnr2_r1 / lambda_ref + lnr3_r2 / lambda_ins) * (T_cg[i] - T_kwi)
                # length-specific heat flow flue gas to outer kiln wall
                q_fg_ko = 1 / (1 / (alpha_fg[i] * r1) + lnr2_r1 / lambda_ref + lnr3_r2 / lambda_ins) * (T_fg[i] - T_kwi)
                # length-specific heat flow outer kiln wall to environment
                # trying to avoid possible overflow by using (a**4 - b**4) = (a**2 + b**2) * (a + b) * (a - b)
                q_ko_amb = alpha_amb * r3 * (T_kwi - T_amb) \
                        + egrad * sigma_sb * r3 * (T_kwi**2 + T_amb**2) * (T_kwi + T_amb) * (T_kwi - T_amb)
                return q_cg_ko + q_fg_ko - 2 * q_ko_amb
            T_kw1 = np.full(z.shape, np.nan)
            for i in range(z.size):
                T_kw1[i] = optimize.brentq(equations2, T_amb, max(T_cg[i], T_fg[i]), args=(i,))
            results['T_kw1'] = T_kw1
            
            dotq_cg_s = alpha_cg_s * A_k * (1 - phi) * s * (T_cg - T_s)
            dotq_fg_s = alpha_fg_s * A_k * (1 - phi) * s * (T_fg - T_s)
            dotH_CO2 = dX_dz * dotM_ls_in / 100 * 44 * h_cg_Ts[:, 2]

            # Energy balance of combustion gas temperature
            dMi_dz = np.outer(dXf, nu_comb) * N_CH40 * barM_g + np.outer(dX_dz, nu_calc) * dotM_ls_in / 100 * barM_g
            q_w = 2 * np.pi / (1 / (alpha_cg * r1) + lnr2_r1 / lambda_ref + lnr3_r2 / lambda_ins) * (T_cg - T_kw1)
            dTcg_dz = (-np.sum(dMi_dz * h_cg_Tcg, axis=-1) + dotH_CO2 - dotq_cg_s - q_w) / np.sum(M_Gi * cp_cg_Tcg, axis=-1)

            # Energy balance of flue gas temperature
            dMi_dz = -np.outer(dX_dz, nu_calc) * dotM_ls_in / 100 * barM_g
            q_w = 2 * np.pi / (1 / (alpha_fg * r1) + lnr2_r1 / lambda_ref + lnr3_r2 / lambda_ins) * (T_fg - T_kw1)
            dTfg_dz = (-np.sum(dMi_dz * h_fg_Tfg, axis=-1) - dotH_CO2 + dotq_fg_s + q_w) / np.sum(M_fg * cp_fg_Tfg, axis=-1)
            
            # Energy balance of solid phase temperature
            dMs_dz = np.outer(dX_dz, np.array([-1, 1])) * dotM_ls_in / 100 * barM_s
            dTs_dz = (- 2 * np.sum(dMs_dz * get_specific_enthalpy_solids(T_s), axis=-1) - 2 * dotH_CO2 + dotq_cg_s + dotq_fg_s) /\
                    np.sum(2 * M_s * cp_s_Ts, axis=-1)

            # Calculation of pressure drops
            dpcg_dz = -get_pressure_drop(p_cg, T_cg, N_cgi)
            dpfg_dz = get_pressure_drop(p_fg, T_fg, N_fg_i)
            
            dy_dz = np.vstack((dX_dz, dTcg_dz, dTs_dz, dTfg_dz, dpcg_dz, dpfg_dz))
            return dy_dz

        def bc(ya, yb):
            return np.array([ya[0] - 0, ya[1] - T_amb, ya[2] - T_amb, yb[3] - T_g_mix, yb[4] - yb[5], ya[5] - p_amb])

        def bc_jac(ya, yb):
            # Providing the Jacobian of the boundary conditions will not save significant time, but it's easy, so let's do it
            dbc_dya = [[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0 ,0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1]]
            dbc_dyb = [[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, -1],
                       [0, 0, 0, 0, 0, 0]]
            return dbc_dya, dbc_dyb
        
        T_history = np.array(sorted(history.keys()))
        if T_history.size >= 2 and T_history.min() <= T_g_mix <= T_history.max():
            # guess initial value by interpolation of results for other temperatures
            y_history = np.array([history[T_i] for T_i in T_history])
            y0 = PchipInterpolator(T_history, y_history)(T_g_mix)
        else:
            # guess linear profiles for conversion, temperatures, profiles and do iterations on coarse grid without grind refinement
            X0, _ = get_Xf(z0)
            X0 *= 0.9
            T_cg0 = np.interp(z0, (z0[0], z0[-1]), (T_amb, T_g_mix+25))
            T_s0 = np.interp(z0, (z0[0], z0[-1]), (T_amb, T_g_mix-25))
            T_fg0 = np.interp(z0, (z0[0], z0[-1]), (T_amb+25, T_g_mix))
            dp = get_pressure_drop(p_amb, T_cg0.mean(), N_cgi0+np.array([0, 0, 1, 0, 0])*dotM_ls_in/100) * (z0[-1] - z0[0])
            dp = dp.squeeze()
            p_cg0 = p_amb + dp + np.interp(z0, (z0[0], z0[-1]), (dp, 0))
            p_fg0 = p_amb + np.interp(z0, (z0[0], z0[-1]), (0, dp))
            y0 = np.array([X0, T_cg0, T_s0, T_fg0, p_cg0, p_fg0])

        sol = optimize.OptimizeResult(x=z0, y=y0)
        for _ in range(30):
            sol = solve_bvp(fun, bc, sol.x, sol.y, bc_jac=bc_jac, max_nodes=max_nodes, tol=math.sqrt(sys.float_info.max), verbose=0)
            if sol.rms_residuals.max() < 1e-2:
                break
        (sol.x, sol.y) = (z0_fine, PchipInterpolator(sol.x, sol.y, axis=1)(z0_fine))
        for _ in range(15):
            sol = solve_bvp(fun, bc, sol.x, sol.y, bc_jac=bc_jac, max_nodes=max_nodes, tol=math.sqrt(sys.float_info.max), verbose=0)
            if sol.rms_residuals.max() < 1e-2:
                break
        # refine if necessary
        tol = 2e-4
        if sol.rms_residuals.max() > tol:
            sol = solve_bvp(fun, bc, z0_fine, sol.y, bc_jac=bc_jac, max_nodes=max_nodes, tol=tol, verbose=0)
        results['part1'] = sol
        if results['part1'].success:
            history[T_g_mix] = results['part1'].sol(z0)

        # Variablen/ Konstanten für Kühlungsabschnitt
        X_end = min(results['part1'].y[0, -1], 1)
        p_cg_end = results['part1'].y[-1, -1]
        T_s_end = results['part1'].y[2, -1]
        M_sc0 = np.array([1 - X_end, X_end]) * dotM_ls_in / 100 * barM_s
        x_a = N_Ai0 / np.sum(N_Ai0)

        def fun_cool(z, y):
            # Definition der Variablen
            T_a = y[0, :]
            T_s = y[1, :]
            p_a = y[2, :]

            # Berechnung des Wärmeübergangskoeffizienten zwischen Kühlluft und Feststoff
            Nu = Nusselt_bed(p_a, T_a, N_Ai0)
            alpha_a = get_thermal_conductivity(p_a, T_a, x_a) * Nu / d_p
            alpha_as = 1 / (1 / alpha_a + d_p / (2 * 5 * lambda_ls))

            # Berechnung Wandwärmeverlust
            def equations2(T_kwi, i):
                # length-specific heat flow cooling air to outer kiln wall
                q_a_ko = 1 / (1 / (alpha_a[i] * r1) + lnr2_r1 / lambda_ref + lnr3_r2 / lambda_ins) * (T_a[i] - T_kwi)
                # length-specific heat flow outer kiln wall to environment
                # trying to avoid possible overflow by using (a**4 - b**4) = (a**2 + b**2) * (a + b) * (a - b)
                q_ko_amb = alpha_amb * r3 * (T_kwi - T_amb) \
                        + egrad * sigma_sb * r3 * (T_kwi**2 + T_amb**2) * (T_kwi + T_amb) * (T_kwi - T_amb)
                # trying to avoid possible overflow by using (a**4 - b**4) = (a**2 + b**2) * (a + b) * (a - b)
                return q_a_ko - q_ko_amb

            T_kw2 = np.full(z.shape, np.nan)
            for i in range(z.size):
                T_kw2[i] = optimize.brentq(equations2, T_amb, T_a[i], args=(i,))
            results['T_kw2'] = T_kw2

            dotq_w = 2 * np.pi / (1 / (alpha_a * r1) + lnr2_r1 / lambda_ref + lnr3_r2 / lambda_ins) * (T_a - T_kw2)
            dotq_a_s = alpha_as * A_k * s * (1 - phi) * (T_a - T_s)

            # Berechnung Ableitung der Temperatur beider Phasen
            cp_a_Ta = get_specific_heat_capacity_gases(T_a)
            dTa_dz = (dotq_a_s + dotq_w) / np.sum(M_Ai0 * cp_a_Ta, axis=-1)
            
            cp_s_Ts = get_specific_heat_capacity_solids(T_s)
            dTs_dz = dotq_a_s / np.sum(M_sc0 * cp_s_Ts, axis=-1)

            # calculation of pressure drop
            dpa_dz = get_pressure_drop(p_a, T_a, N_Ai0)

            return np.vstack((dTa_dz, dTs_dz, dpa_dz))

        def bc_cool(ya, yb):
            return np.array([yb[0] - T_amb, ya[1] - T_s_end, ya[2] - p_cg_end])
        
        def bc_cool_jac(ya, yb):
            dbc_dya = [[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]
            dbc_dyb = [[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]
            return dbc_dya, dbc_dyb

        T_s0 = np.linspace(T_s_end, T_amb+5, z0_cool.size)
        T_a0 = np.linspace(T_s_end-5, T_amb, z0_cool.size)
        p_a0 = np.full(z0_cool.shape, p_cg_end)
        y0_cool = np.array([T_a0, T_s0, p_a0])
        
        results['part2'] = solve_bvp(fun_cool, bc_cool, z0_cool, y0_cool, bc_jac=bc_cool_jac, tol=tol*1e-1, verbose=0)

        # calculate error of energy balance error combustion gas + 2 * cooling air = flue gas
        Xf_end = get_Xf(results['part1'].x)[0][-1]
        T_cg_end = results['part1'].y[1, -1]
        N_cgi_out = N_cgi0 + Xf_end * N_CH40 * nu_comb + X_end * dotM_ls_in / 100 * nu_calc
        M_cgi_out = N_cgi_out * barM_g
        h_cg_out = get_specific_enthalpy_gases(T_cg_end)

        T_a_out = results['part2'].y[0, 0]
        h_a_out = get_specific_enthalpy_gases(T_a_out)

        M_fg_in = M_cgi_out + 2 * M_Ai0
        h_fg_in = get_specific_enthalpy_gases(T_g_mix)

        balance_error =  np.sum(2 * M_Ai0 * h_a_out) + np.sum(M_cgi_out * h_cg_out) - np.sum(M_fg_in * h_fg_in)

        errorFun_kiln.success = results['part1'].success and results['part2'].success
        errorFun_kiln.fun = fun  # make it callable from outside
        errorFun_kiln.fun_cool = fun_cool  # dito
        # print(f'{T_g_mix = }, {balance_error = }')
        return balance_error
    
    errorFun_kiln.history = dict()
    T_low = get_initial_value(dotM_ls_in, dotM_CH4, dotM_airCombustion, dotM_airCool_in, 1) - 100
    T_high = get_initial_value(dotM_ls_in, dotM_CH4, dotM_airCombustion, dotM_airCool_in, 0) + 200
    T_mix, r = optimize.brentq(errorFun_kiln, T_low, T_high, xtol=1e-8, rtol=1e-8, maxiter=20, disp=False, full_output=True)
    results['n_iter'] = r.iterations
    balance_error = errorFun_kiln(T_mix)  # entries of results is filled during execution
    errorFun_kiln.fun(results['part1'].x, results['part1'].y)  # dito
    errorFun_kiln.fun_cool(results['part2'].x, results['part2'].y)  # dito

    # calculate temperature in crossover channel
    Xf_end = get_Xf(results['part1'].x)[0][-1]
    X_end = min(results['part1'].y[0, -1], 1)
    N_cgi_out = N_cgi0 + Xf_end * N_CH40 * nu_comb + X_end * dotM_ls_in / 100 * nu_calc
    M_cgi_out = N_cgi_out * barM_g
    T_cg_end = results['part1'].y[1, -1]
    h_cg_out = get_specific_enthalpy_gases(T_cg_end)
    T_a_out = results['part2'].y[0, 0]
    h_a_out = get_specific_enthalpy_gases(T_a_out)
    def equations3(T_crossover):
        h_co = get_specific_enthalpy_gases(T_crossover)
        error = np.sum(M_Ai0 * h_a_out) + np.sum(M_cgi_out * h_cg_out) - np.sum((M_cgi_out + M_Ai0) * h_co)
        return error
    
    # results['T_co'] = optimize.brentq(equations3, T_a_out, T_cg_end)
    results['T_co'] = optimize.brentq(equations3, T_a_out, T_cg_end)

    # ### CHECK OVERALL BALANCES ###
    # M_sc0 = np.array([1 - X_end, X_end]) * dotM_ls_in / 100 * barM_s
    # N_fg_out = N_cgi0 + 2 * N_Ai0 + Xf_end * nu_comb * N_CH40 \
    #                             + 2 * X_end * nu_calc * dotM_ls_in / 100
    # M_fg_out = N_fg_out * barM_g
    # # mass balance
    # dotM_in = 2 * dotM_ls_in + dotM_CH4_in + dotM_airCombustion_in + 2 * dotM_airCool_in
    # dotM_out = 2 * M_sc0.sum() + M_fg_out.sum()
    # print(dotM_out / dotM_in - 1)

    # # energy balance
    # dotH_in = 2 * get_specific_enthalpy_solids(T_amb)[0] * dotM_ls_in + (M_Gi0 * get_specific_enthalpy_gases(T_amb)).sum() + 2 * (M_Ai0 * hgasint(T_amb)).sum()
    # T_s_out = results['part2'].y[1, -1]
    # T_fg_out = results['part1'].y[3, 0]
    # dotH_out = 2 * (M_sc0 * get_specific_enthalpy_solids(T_s_out)).sum() + (M_fg_out * get_specific_enthalpy_gases(T_fg_out)).sum()
    # # heat loss
    # z = np.concatenate((results['part1'].x, results['part2'].x))
    # T_kw = np.concatenate((results['T_kw1'], results['T_kw2']))
    # dotq_kw = 2 * np.pi * r3 * (alpha_amb * (T_kw - T_amb) + egrad * sigma_sb * (T_kw**4 - T_amb**4))
    # dotQ_out = 2 * trapezoid(dotq_kw, z)  # 2 shafts
    # # print((dotH_out + dotQ_out) / dotH_in - 1)
    # dotC = 2 * (M_sc0 * cpfestint(T_s_out)).sum() + (M_fg_out * cpgasint(T_fg_out)).sum()  # heat capacity stream
    # print((dotH_in - dotH_out - dotQ_out) / dotC)

    convergence_energy_balance = abs(balance_error / (dotM_CH4 * get_heat_of_combustion(T0))) < 1e-6
    if errorFun_kiln.success and convergence_energy_balance and r.converged:
        return results
    else:
        error_msg = (
            f"Model did not converge. The convergence criteria are:\n"
            f"\tOuter loop: {r.converged}\n"
            f"\tInner loops: {errorFun_kiln.success}\n"
            f"\tEnergy balance: {convergence_energy_balance}"
        )
        raise Exception(error_msg)


def get_initial_value(dotM_ls_in, dotM_CH4, dotM_airCombustion, dotM_airCool_in, X_guess):
    ## assume thermal equilibrium and total conversion and fuel burning
    ## to get an initial guess for T_gmix
    # balance around burning shaft and both cooling zones
    # entering streams:
    # - limestone feed (burning shaft) at ambient temperature
    # - quicklime product (non-burning shaft) at temperature tbd
    # - fuel
    # - combustion air
    # - cooling air
    # leaving streams:
    # - quicklime product (both shafts) at ambient temperature
    # - flue gas at temperature tbd

    dotM_gas_in = w_air * (dotM_airCombustion + 2*dotM_airCool_in) + np.array([0, 0, 0, 0, 1]) * dotM_CH4
    h_gas_in = get_specific_enthalpy_gases(T_amb)
    dotH_gas_in = np.sum(dotM_gas_in * h_gas_in)

    dotM_gas_out = dotM_gas_in \
                   + nu_comb * dotM_CH4 / 16 * barM_g \
                   + X_guess * dotM_ls_in / 100 * 44 * np.array([0, 0, 1, 0, 0])

    dotM_solid_out = 2 * dotM_ls_in * (np.array([1, 0]) + X_guess * np.array([-1, 1]) * barM_s / 100)
    h_solid_out = get_specific_enthalpy_solids(T_amb)
    dotH_solid_out = np.sum(dotM_solid_out * h_solid_out)

    dotM_solid_in_burn = dotM_ls_in * np.array([1, 0])
    h_solid_in_burn = get_specific_enthalpy_solids(T_amb)
    dotH_solid_in_burn = np.sum(dotM_solid_in_burn * h_solid_in_burn)

    dotM_solid_in_nonburn = dotM_ls_in * (np.array([1, 0]) + X_guess / 100 * barM_s * np.array([-1, 1]))

    dotM_in = dotM_solid_in_burn.sum() + dotM_solid_in_nonburn.sum() + dotM_gas_in.sum()
    dotM_out = dotM_gas_out.sum() + dotM_solid_out.sum()
    if abs(dotM_in / dotM_out - 1) > 1e-10:
        raise Exception('Something is wrong in the mass balance')

    def balance_error(T):
        h_gas_out = get_specific_enthalpy_gases(T)
        dotH_gas_out = np.sum(dotM_gas_out * h_gas_out)

        h_solid_in_nonburn = get_specific_enthalpy_solids(T)
        dotH_solid_in_nonburn = np.sum(dotM_solid_in_nonburn * h_solid_in_nonburn)
        dotH_solid_in = dotH_solid_in_burn + dotH_solid_in_nonburn
        return dotH_gas_in + dotH_solid_in - dotH_gas_out - dotH_solid_out

    # Just an initial value, so no high tolerances needed...
    T_g_mix0 = optimize.brentq(balance_error, T_amb, 2500, xtol=1e-4, rtol=1e-4)
    return T_g_mix0


if __name__ == "__main__":
    # k_CH4 = 0.95022858 
    # k_ls = 0.90064462
    # dotM_ls_in *= k_ls
    # dotM_CH4 *= k_CH4
    # dotM_airCombustion *= k_CH4
    # dotM_airCool_in *= k_ls

    res = calculateKiln(dotM_ls_in, dotM_CH4, dotM_airCombustion, dotM_airCool_in)
    if res:
        sol = res['part1']
        sol_cool = res['part2']
        sol_df = pd.DataFrame(sol.y.T, index=sol.x,
                            columns=('X', 'T_cg/C', 'T_s/C', 'T_fg/C', 'p_cg/mbar', 'p_fg/mbar'))
        sol_df.loc[:, ('T_cg/C', 'T_s/C', 'T_fg/C')] -= 273.15
        sol_df.loc[:, ('p_cg/mbar', 'p_fg/mbar')] *= 1e-2
        sol_df.loc[:, 'X_f'] = get_Xf(sol.x)[0]
        for i, comp in enumerate(('N2', 'O2', 'CO2', 'H2O', 'CH4')):
            sol_df.loc[:, f'p_{comp}_cg/mbar'] = sol_df.loc[:, 'p_cg/mbar'] * res['x_cg'][:, i]
        sol_cool_df = pd.DataFrame(sol_cool.y.T, index=sol_cool.x, columns=('T_a/C', 'T_s/C', 'p_a/mbar'))
        sol_cool_df.loc[:, ('T_a/C', 'T_s/C')] -= 273.15
        sol_cool_df.loc[:, ('p_a/mbar')] *= 1e-2
        solution = sol_df.combine_first(sol_cool_df)
        solution.index.name = 'z'
        X = sol.y[0, -1]
        fuel_efficiency = dotM_ls_in * 56 / 100 * X / dotM_CH4
        deltap = sol_df.at[0, 'p_cg/mbar'] - p_amb * 1e-2
        print(f'{X=:.3f}, {fuel_efficiency=:.2f}, {deltap=:.2f} mbar')
        solution.plot(subplots=[['X', 'X_f'],
                                [f'T_{i}/C' for i in ('cg', 'fg', 'a', 's')],
                                [f'p_{i}/mbar' for i in ('cg', 'fg', 'a')],
                                [f'p_{i}_cg/mbar' for i in ('N2', 'O2', 'CO2', 'H2O', 'CH4')]])
        plt.show()
        # save on reasonably fine grid
        z_csv = np.linspace(0, L_p+L_r+L_c, int((L_p+L_r+L_c)/0.1)+1)
        solution.reindex(solution.index.union(z_csv)).interpolate('index', limit_area='inside').reindex(z_csv).to_csv('out/kiln_profiles_initial.csv')
    else:
        print('Kiln model did not converge.')