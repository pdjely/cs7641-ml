import numpy as np
import pandas as pd
import mlrose
from . import util


def fourpeaks(max_iter=500, early_stop=None,
              mimic_early_stop=100, savedir=None):
    print('\n|========= Four Peaks =========|\n\n')
    fitness = mlrose.FourPeaks(t_pct=0.10)
    problem_size = [30, 60, 90]
    max_attempts = max_iter * 2 if early_stop is None else early_stop
    mimic_early_stop = max_attempts if mimic_early_stop is None else mimic_early_stop
    hyperparams = {
        'rhc': {
            'restarts': 0,
            'max_attempts': max_attempts
        },
        'mimic': {
            'pop_size': 3000,
            'keep_pct': 0.15,
            'max_attempts': mimic_early_stop,
            'fast_mimic': True
        },
        'sa': {
            'schedule': mlrose.GeomDecay(),
            'init_state': None,
            'max_attempts': max_attempts
        },
        'ga': {
            'pop_size': 200,
            'mutation_prob': 0.2,
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

        r, t, wt = util.optimize_iters(problem, max_iter, hyperparams)
        print('Last five fitness scores: ')
        print(r.tail(5), '\n')
        results.append(r)
        runtimes.append(t)
        timings['ps{}'.format(ps)] = wt

    print('final runtimes')
    t = pd.DataFrame(runtimes, index=problem_size)
    print(t)

    if savedir:
        t.to_csv('{}/fourpeaks_runtimes.csv'.format(savedir))
        for i, df in enumerate(results):
            df.to_csv('{}/fourpeaks_ps{}.csv'.format(savedir, problem_size[i]))
        # Write the timings as a single dataframe
        for k, v in timings.items():
            for atype, times in v.items():
                tdf = pd.DataFrame(times, columns=['time', 'fitness'])
                tdf.to_csv('{}/fourpeaks_{}_{}_timings.csv'
                           .format(savedir, k, atype))

    return t, results, timings
