import mlrose
import numpy as np
import pandas as pd
import timeit


def optimize_iters(problem, max_iters, hyperparams):
    """
    Run all four optimization algorithms on given problem for n iters

    :param max_iters: int maximum iterations
    :param problem: mlrose optimization problem
    :param hyperparams: dict containing hyperparams for the four algos
    :return: pandas dataframe with observations in rows and algo fitness
        in columns
    """
    print('Max iterations: {}'.format(max_iters))
    random_state = 10

    # Set up the four optimization algorithms
    algos = [mlrose.random_hill_climb, mlrose.mimic,
             mlrose.simulated_annealing, mlrose.genetic_alg]
    algo_names = ['rhc', 'mimic', 'sa', 'ga']
    algo_long_name = {
        'rhc': 'Random Hill Climbing',
        'mimic': 'MIMIC',
        'sa': 'Simulated Annealing',
        'ga': 'Genetic Algorithm'
    }

    # TODO: Run 5 times and average results
    results = {}
    runtimes = {}
    timings = {}
    for algo, name in zip(algos, algo_names):
        # Timing from StackOverflow
        # Source:
        print('Running {}'.format(algo_long_name[name]))
        start_time = timeit.default_timer()
        print(hyperparams[name])
        best_state, best_fitness, curve = algo(problem,
                                               max_iters=max_iters,
                                               curve=True,
                                               timing=True,
                                               random_state=random_state,
                                               **hyperparams[name])
        end_time = timeit.default_timer()
        # Take just the fitness values from curve for plotting iterations-fitness
        results[name] = curve[:, 1]
        runtimes[name] = end_time - start_time
        timings[name] = curve

    # DataFrame from uneven lists
    # https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    df.index.name = 'iter'
    return df, runtimes, timings
