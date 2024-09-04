import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from limekiln import calculateKiln
from limekiln.auxilaries import dotM_ls_in, dotM_CH4, dotM_airCombustion, dotM_airCool_in, get_Xf, p_amb, L_p, L_r, L_c

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