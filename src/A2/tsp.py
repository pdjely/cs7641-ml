import numpy as np
import mlrose
import pandas as pd
from . import util


def get_tsp_coords(n, max_coord=200, random_state=None):
    """
    Generate traveling salesperson problem coordinates

    Creates n (x, y) coordinate pairs for the traveling salesperson taken from
    an array of max_coord length without replacement.

    :param n: int, number of (x, y) tuples to generate
    :param max_coord: int, maximum number to sample from
    :param random_state: int, seed for random number generator
    :return: n x 2 ndarray


    """
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.choice(max_coord, (n, 2), replace=False)


def tsp(max_iter=500, early_stop=None, mimic_early_stop=100, n_runs=10,
        savedir=None):
    """
    Travelling Salesperson Problem

    Steps to solve TSP in mlrose:
      1. Define a fitness function object
      2. Define an optimization problem object
      3. Select and run a randomized optimization algorithm

    Source: mlrose documentation
    https://mlrose.readthedocs.io/en/stable/source/tutorial2.html

    :param max_iter:
    :param early_stop:
    :param mimic_early_stop:
    :param n_runs:
    :param savedir:
    :return:
    """
    print('\n\n|========= Traveling Salesman =========|\n')
    problem_size = [10, 30, 60]
    max_attempts = max_iter * 2 if early_stop is None else early_stop
    mimic_early_stop = max_attempts if mimic_early_stop is None else mimic_early_stop
    hyperparams = {
        'rhc': {
            'restarts': 0,
            'max_attempts': max_attempts
        },
        'mimic': {
            'pop_size': 1000,
            'keep_pct': 0.15,
            'max_attempts': mimic_early_stop,
            'fast_mimic': True
        },
        'sa': {
            'schedule': mlrose.ExpDecay(),
            'init_state': None,
            'max_attempts': max_attempts
        },
        'ga': {
            'pop_size': 2000,
            'mutation_prob': 0.3,
            'pop_breed_percent': 0.75,
            'elite_dreg_ratio': 0.85,
            'max_attempts': max_attempts
        }
    }
    print('Hyperparameters: ', hyperparams)

    results = []
    runtimes = []
    timings = {}
    for ps in problem_size:
        print('Running with {} city locations'.format(ps))
        print('------------------------------------')
        coords_list = get_tsp_coords(ps)
        print(coords_list)

        # Set up the problem
        fitness_coords = TravellingSalesMaxFit(coords=coords_list)
        problem = mlrose.TSPOpt(length=ps, fitness_fn=fitness_coords,
                                maximize=True)

        # Solve it
        f, t, wt = util.optimize_iters(problem, max_iter, hyperparams, n_runs)
        results.append(f)
        runtimes.append(t)
        timings['ps{}'.format(ps)] = wt

    print('final runtimes')
    t = pd.DataFrame(runtimes, index=problem_size)
    print(t)

    if savedir:
        util.save_output('tsp', savedir, t, results,
                         timings, problem_size)

    return t, results, timings


class TravellingSalesMaxFit(mlrose.TravellingSales):
    """
    Convert fitness function to be maximized by taking inverse
    """
    def evaluate(self, state):
        fitness = super().evaluate(state)
        return 1 / fitness
