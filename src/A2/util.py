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
    random_state = 112

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
        random_state += 123
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
            if name == 'mimic':
                # print('best fitness: {}'.format(best_fitness))
                print('{} {}: fitness {}'.format(name, i, curve[:, 1]))

            # only save if this is the best run, do this instead of averaging
            # because each run can have different lengths AND there were
            # some wild variations in the MIMIC implementation
            if i == 0 or best_fitness > np.max(timings[name][:, 1]):
                results[name] = curve[:, 1]
                runtimes[name] = end_time - start_time
                timings[name] = curve

    # DataFrame from uneven lists
    # https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    df.index.name = 'iter'

    print('Last five fitness scores: ')
    print(df.tail(5), '\n')
    return df, runtimes, timings


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
