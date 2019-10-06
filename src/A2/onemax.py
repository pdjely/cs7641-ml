import numpy as np
import pandas as pd
import mlrose
from . import util


def onemax(max_iter=500, early_stop=None,
           mimic_early_stop=100, n_runs=10,
           savedir=None):
    print('\n\n|========= One Max =========|\n')
    fitness = mlrose.OneMax()
    problem_size = [10, 100, 1000]
    max_attempts = max_iter * 2 if early_stop is None else early_stop
    mimic_early_stop = max_attempts if mimic_early_stop is None else mimic_early_stop
    hyperparams = {
        'rhc': {
            'restarts': 0,
            'max_attempts': max_attempts
        },
        'mimic': {
            'pop_size': 2000,
            'keep_pct': 0.3,
            'max_attempts': mimic_early_stop,
            'fast_mimic': True
        },
        'sa': {
            'schedule': mlrose.GeomDecay(),
            'init_state': None,
            'max_attempts': max_attempts
        },
        'ga': {
            'pop_size': 2000,
            'mutation_prob': 0.25,
            'pop_breed_percent': 0.60,
            'elite_dreg_ratio': 0.75,
            'max_attempts': max_attempts
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
        util.save_output('onemax', savedir, t, results, timings, problem_size)

    return t, results, timings
