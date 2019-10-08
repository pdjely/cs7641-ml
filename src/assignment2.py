import mlrose
import A2
import util
import argparse
import datajanitor
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def main():
    args = getArgs()
    savedir = util.mktmpdir(args.outdir)

    problem_table = {
        'fourpeaks': fourpeaks,
        'tsp': tsp,
        'flipflop': flipflop,
        'onemax': onemax
    }
    for p in args.problems:
        problem_table[p](savedir)


def run_mlweight():
    """
    Demo neural network weights by random optimization

    Steps:
      1. Initialize a machine learning weight optimization problem object.
      2. Find the optimal model weights for a given training dataset by
         calling the fit method of the object initialized in step 1.
      3. Predict the labels for a test dataset by calling the predict method
         of the object initialized in step 1.

    source:
    https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
    :return:
    """
    dataset = datajanitor.getDataset('musk')
    dataset.getData()

    x_train, x_test, y_train, y_test = dataset.partitionData(percent=0.3,
                                                             randomState=10)
    # data preprocessing
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    # 1. Initial neural net model
    nn_model = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
                                    algorithm='random_hill_climb', max_iters=1000,
                                    bias=True, is_classifier=True, learning_rate=0.0001,
                                    early_stopping=True, clip_max=5,
                                    max_attempts=100, random_state=3)

    nn_model.fit(x_train_scaled, y_train_hot)
    y_train_pred = nn_model.predict(x_train_scaled)
    # Predict labels for train set and assess accuracy
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    print(y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model.predict(x_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    print(y_test_accuracy)


def fourpeaks(savedir=None):
    # With early stopping
    t, r, timings = A2.fourpeaks(max_iter=np.inf,
                                 early_stop=2000,
                                 mimic_early_stop=50,
                                 n_runs=10,
                                 savedir=savedir)


def tsp(savedir=None):
    t, r, timings = A2.tsp(max_iter=1000,
                           early_stop=100,
                           mimic_early_stop=30,
                           n_runs=1,
                           savedir=savedir)


def flipflop(savedir=None):
    t, r, timings = A2.flipflop(max_iter=np.inf,
                                early_stop=2000,
                                mimic_early_stop=50,
                                n_runs=10,
                                savedir=savedir)


def onemax(savedir=None):
    t, r, timings = A2.onemax(max_iter=np.inf,
                              early_stop=1000,
                              mimic_early_stop=10,
                              n_runs=5,
                              savedir=savedir)


def getArgs():
    parser = argparse.ArgumentParser(description='CS7641 Assignment 2')

    validProblems = ['fourpeaks', 'tsp', 'onemax', 'flipflop']

    parser.add_argument('-p', '--problems',
                        help='Space-separated list of problems to run (default: all)',
                        choices=validProblems, default=validProblems,
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
