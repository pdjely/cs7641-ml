import mlrose
import A2
import util
import argparse
import numpy as np


def main():
    args = getArgs()
    savedir = util.mktmpdir(args.outdir)

    problem_table = {
        'fourpeaks': fourpeaks,
        'tsp': tsp,
        'flipflop': flipflop,
        'onemax': onemax,
        'nn': A2.run_mlweight
    }
    for p in args.problems:
        problem_table[p](savedir)


def fourpeaks(savedir=None):
    # With early stopping
    t, r, timings = A2.fourpeaks(max_iter=np.inf,
                                 early_stop=2000,
                                 mimic_early_stop=50,
                                 n_runs=10,
                                 savedir=savedir)


def tsp(savedir=None):
    t, r, timings = A2.tsp(max_iter=np.inf,
                           early_stop=1000,
                           mimic_early_stop=50,
                           n_runs=10,
                           savedir=savedir)


def flipflop(savedir=None):
    t, r, timings = A2.flipflop(max_iter=np.inf,
                                early_stop=200,
                                mimic_early_stop=20,
                                n_runs=1,
                                savedir=savedir)


def onemax(savedir=None):
    t, r, timings = A2.onemax(max_iter=np.inf,
                              early_stop=1000,
                              mimic_early_stop=10,
                              n_runs=5,
                              savedir=savedir)


def getArgs():
    parser = argparse.ArgumentParser(description='CS7641 Assignment 2')

    validProblems = ['fourpeaks', 'tsp', 'onemax', 'flipflop', 'nn']

    parser.add_argument('-p', '--problems',
                        help='Space-separated list of problems to run (default: all)',
                        choices=validProblems, default=['fourpeaks', 'tsp', 'onemax', 'nn'],
                        nargs='+')
    # parser.add_argument('-d', '--datasets',
    #                     help='Space-separated list of datasets (default: all)',
    #                     choices=validData, default=validData,
    #                     nargs='+')
    parser.add_argument('-o', '--outdir',
                        help='Directory to save files to (optional: default timestamp)',
                        nargs='?', default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
