import numpy as np
import pandas as pd
import mlrose
from . import util


def fourpeaks(max_iter=500, early_stop=None, savedir=None):
    # TODO: refactor into separate file under A2
    print('\n|========= Four Peaks =========|\n\n')
    fitness = mlrose.FourPeaks(t_pct=0.10)
    problem_size = [30, 60, 90]
    early_stop = max_iter * 2 if early_stop is None else early_stop
    hyperparams = {
        'rhc': {
            'restarts': 0,
            'max_attempts': early_stop
        },
        'mimic': {
            'pop_size': 3000,
            'keep_pct': 0.15,
            'max_attempts': 10
        },
        'sa': {
            'schedule': mlrose.GeomDecay(),
            'init_state': None,
            'max_attempts': early_stop
        },
        'ga': {
            'pop_size': 200,
            'mutation_prob': 0.2,
            'max_attempts': early_stop
        }
    }

    results = []
    timings = []
    for ps in problem_size:
        problem = mlrose.DiscreteOpt(ps, fitness, max_val=2, maximize=True)
        print('Running with input size', ps)
        print('-----------------------------')

        r, t = util.optimize_iters(problem, max_iter, hyperparams)
        print('Last five fitness scores: ')
        print(r.tail(5), '\n')
        results.append(r)
        timings.append(t)

    print('final timings')
    t = pd.DataFrame(timings, index=problem_size)
    print(t)

    if savedir:
        t.to_csv('{}/timings.csv'.format(savedir))
        for i, df in enumerate(results):
            df.to_csv('{}/iter{}.csv'.format(savedir, i))

    return t, results
