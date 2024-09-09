import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import pygad
from scipy.optimize import Bounds, NonlinearConstraint, minimize, SR1
from limekiln.model import calculateKiln, dotM_airCombustion, dotM_airCool_in, dotM_CH4, dotM_ls_in
from functools import lru_cache
from datetime import datetime


def optimize_geneticAlgorithm():
    with open('out/ann.pkl', 'rb') as fh:
        ann_dict = pickle.load(fh)
    ann = ann_dict['ann']
    xScaler = ann_dict['xScaler']
    yScaler = ann_dict['yScaler']


    # batch solution input with two genes for multi-object optimization
    def fitness_function_batch(ga_instance, solution, solution_idx):
        k_CH4 = solution[:,0]
        k_ls = solution[:,1]
        x = np.array([k_ls * dotM_ls_in, k_CH4 * dotM_airCombustion, k_CH4 * dotM_CH4, k_ls * dotM_airCool_in]).T
        x_scaled = xScaler.transform(x)
        y_scaled = ann.predict(x_scaled, verbose=0)
        y = yScaler.inverse_transform(y_scaled)
        X = y[:,0]
        X[X > 1] = 1
        # objective 1: high conversion / low residual CO2 content
        objective1 = - (X - 0.98)**2
        # objective 2: high quicklime production per fuel consumption
        dotM_ql = k_ls * dotM_ls_in / 100 * X * 56
        objective2 = dotM_ql / (k_CH4 * dotM_CH4)  # specific fuel consumption
        return np.array([objective1, objective2]).T  # had to change PyGAD source code to make this work (bug?)

    
    fitnessHistory = pd.DataFrame()
    def save_fitness(ga_instance, fitness):
        nonlocal fitnessHistory
        for i in range(fitness.shape[-1]):
            colName = f'gen{ga_instance.generations_completed}_obj{i}'
            fitnessHistory.loc[:, colName] = fitness[:, i]

    def plot_fitness(ga_instance, fitness):
        if not hasattr(plot_fitness, 'ax'):
            _, plot_fitness.ax = plt.subplots(1, 1)
        plot_fitness.ax.scatter(fitness[:, 0], fitness[:, 1], label=ga_instance.generations_completed)

    def on_fitness(ga_instance, population_fitness):
        # store every 25th population
        if ga.generations_completed % 25:
            return
        save_fitness(ga_instance, population_fitness)
        plot_fitness(ga_instance, population_fitness)

    def on_stop(ga_instance, last_population_fitness):
        save_fitness(ga_instance, last_population_fitness)
        plot_fitness(ga_instance, last_population_fitness)

    
    gene_space = {'low': 0.9, 'high': 1.1}
    population_size = 100

    ga = pygad.GA(num_generations=200,
                num_genes=2,
                gene_space=gene_space,
                fitness_func=fitness_function_batch,
                fitness_batch_size=population_size,
                sol_per_pop=population_size,
                num_parents_mating=15,
                parent_selection_type='tournament_nsga2',
                keep_elitism=5,
                mutation_probability=0.20,
                # keep_parents=2,
                on_fitness=on_fitness,
                on_stop=on_stop,
                crossover_probability=0.10)
                # save_solutions=True)
    t1 = datetime.now()
    ga.run()
    t2 = datetime.now()
    deltat = t2 - t1
    print(f'Execution time of GA: {deltat.seconds + deltat.microseconds/1e6} s')
    plt.legend()
    plt.show()
    ga.summary()
    # ga.plot_fitness()
    # sol, sol_fitness, _ = ga.best_solution()
    # fitness.index.name = ''
    fitnessHistory.to_csv('out/optimization_fitness_history.csv', index=False)
    solution = pd.DataFrame(ga.population, columns=['k_CH4', 'k_ls'])
    solution.to_csv('out/optimization_population.csv', index=False)
    pass

def optimize_numeric():

    @lru_cache
    def runModel(k):
        k_CH4, k_ls = k
        result = calculateKiln(k_ls * dotM_ls_in, k_CH4 * dotM_CH4, k_CH4 * dotM_airCombustion, k_ls * dotM_airCool_in)
        X = result['part1'].y[0, -1]
        fuel_efficiency = (k_ls * dotM_ls_in) * 56 / 100 * X / (k_CH4 * dotM_CH4)
        return (X, fuel_efficiency)

    def get_objective(k):
        k_tuple = tuple(k)  # make hashable for lru_cache
        _, fuel_efficiency = runModel(k_tuple)
        return 1 / fuel_efficiency
    
    def get_conversion_constraint(k):
        k_tuple = tuple(k)  # make hashable for lru_cache
        X, _ = runModel(k_tuple)
        return X - 0.98

    constraint_conversion = NonlinearConstraint(get_conversion_constraint, 0, 0, jac='2-point', hess=SR1())

    k0 = (1, 1)
    bounds = Bounds([0.9, 0.9], [1.1, 1.1])
    options = {'gtol':1e-6, 'xtol': 1e-6, 'verbose': 3, 'initial_tr_radius': 1e0}
    res = minimize(get_objective, k0, method='trust-constr',  jac='2-point', hess=SR1(),
            constraints=[constraint_conversion], options=options, bounds=bounds)
    
    now = datetime.now()
    filename = f"out/{now.strftime('%Y-%m-%d-%H-%M-%S')}_optimizationResult.txt"
    with open(filename, 'w') as fh:
        fh.write(str(res)+"\n")
        fh.write(f'x0 = {k0}\n')
        fh.write(f'x = {str(res.x)}')
    

if __name__ == '__main__':
    # optimize_geneticAlgorithm()
    optimize_numeric()