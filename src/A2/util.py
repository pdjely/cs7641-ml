import mlrose
import numpy as np
import pandas as pd
import timeit


def optimize_iters(problem, max_iters, hyperparams, n_runs=10):
    """
    Run all four optimization algorithms on given problem for n iters

    :param max_iters: int maximum iterations
    :param problem: mlrose optimization problem
    :param hyperparams: dict containing hyperparams for the four algos
    :param n_runs: number of runs to average results over
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

    results = {}
    runtimes = {}
    timings = {}
    print('Algorithm hyperparameters: ', hyperparams)
    for i in range(n_runs):
        random_state += 10
        print('Starting run {}'.format(i))
        for algo, name in zip(algos, algo_names):
            start_time = timeit.default_timer()
            best_state, best_fitness, curve = algo(problem,
                                                   max_iters=max_iters,
                                                   curve=True,
                                                   timing=True,
                                                   random_state=random_state,
                                                   **hyperparams[name])
            end_time = timeit.default_timer()
            # Take just the fitness values from curve for plotting iterations-fitness
            if i == 0:
                results[name] = curve[:, 1]
                runtimes[name] = end_time - start_time
                timings[name] = curve
            else:
                results[name] = add_diffsize(results[name], curve[:, 1])
                runtimes[name] = runtimes[name] + end_time - start_time
                # timings independent variable is wall clock time, which makes
                # it non-trivial to average. So instead just take the timing
                # with the highest fitness
                print('max iteration fitness: ', np.max(curve[:, 1]))
                print('max saved fitness: ', np.max(timings[name][:, 1]))
                if np.max(curve[:, 1]) > np.max(timings[name][:, 1]):
                    timings[name] = curve

    for k in results.keys():
        results[k] = results[k] / n_runs
    for k in runtimes.keys():
        runtimes[k] = runtimes[k] / n_runs
    # DataFrame from uneven lists
    # https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    df.index.name = 'iter'

    print('Last five fitness scores: ')
    print(df.tail(5), '\n')
    return df, runtimes, timings


def add_diffsize(a, b):
    """
    Add two numpy arrays of possibly different size

    Stolen word for word from Stack Overflow:
        https://stackoverflow.com/questions/7891697/numpy-adding-two-vectors-with-different-sizes
    :param a: numpy array, first operand
    :param b: numpy array, second operand
    :return: numpy array, newly allocated to length of longest of a and b
    """
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
        c[len(a):] += a[-1]
    else:
        c = a.copy()
        c[:len(b)] += b
        c[len(b):] += b[-1]
    return c


def save_output(problem_name, savedir, runtimes,
                results, timings, problem_size):
    """
    Save optimization results to output directory

    :param problem_name: string, name of optimization problem
    :param savedir: string, output directory
    :param runtimes: dataframe, total runtimes by problem_size
    :param results: dataframe, fitness scores by iteration
    :param timings: dataframe, fitness scores by wall clock time
    :param problem_size: list, input sizes
    """
    runtimes.to_csv('{}/{}_runtimes.csv'.format(savedir, problem_name))
    for i, df in enumerate(results):
        df.to_csv('{}/{}_ps{}.csv'.format(savedir, problem_name,
                                          problem_size[i]))

    # Write the timings as a single dataframe
    for k, v in timings.items():
        for atype, times in v.items():
            tdf = pd.DataFrame(times, columns=['time', 'fitness'])
            tdf.to_csv('{}/{}_{}_{}_timings.csv'
                       .format(savedir, problem_name, k, atype))
