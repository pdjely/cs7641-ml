import numpy as np
import pandas as pd
import mlrose
from . import util


def queens(max_iter=500, early_stop=None,
             mimic_early_stop=100, n_runs=10,
             savedir=None):
    print('\n\n|========= N Queens =========|\n')
    fitness = mlrose.CustomFitness(queens_max)
    problem_size = [8, 64, 128]
    max_attempts = max_iter * 2 if early_stop is None else early_stop
    mimic_early_stop = max_attempts if mimic_early_stop is None else mimic_early_stop
    hyperparams = {
        'rhc': {
            'restarts': 0,
            'max_attempts': max_attempts
        },
        'mimic': {
            'pop_size': 500,
            'keep_pct': 0.1,
            'max_attempts': mimic_early_stop,
            'fast_mimic': True
        },
        'sa': {
            'schedule': mlrose.GeomDecay(),
            'init_state': None,
            'max_attempts': max_attempts
        },
        'ga': {
            'pop_size': 500,
            'mutation_prob': 0.2,
            'pop_breed_percent': 0.6,
            'elite_dreg_ratio': 0.95,
            'max_attempts': mimic_early_stop
        }
    }
    print('Hyperparameters: ', hyperparams)

    results = []
    runtimes = []
    timings = {}
    for ps in problem_size:
        problem = mlrose.DiscreteOpt(ps, fitness, max_val=2, maximize=True)
        print('Running with input size', ps)
        print('-----------------------------')

        r, t, wt = util.optimize_iters(problem, max_iter, hyperparams, n_runs)
        results.append(r)
        runtimes.append(t)
        timings['ps{}'.format(ps)] = wt

    print('final runtimes')
    t = pd.DataFrame(runtimes, index=problem_size)
    print(t)

    if savedir:
        util.save_output('flipflop', savedir, t, results, timings, problem_size)

    return t, results, timings


# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    """
    Maximization fitness function for queens
    
    Source:
    https://mlrose.readthedocs.io/en/stable/source/tutorial1.html#select-and-run-a-randomized-optimization-algorithm
    :param state: 
    :return: 
    """
    # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):

                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt
