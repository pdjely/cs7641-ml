import mlrose
import A2
import util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run_tsp():
    """
    Travelling Salesperson Problem Example

    Example of steps to solve an optimization problem in mlrose
      1. Define a fitness function object
      2. Define an optimization problem object
      3. Select and run a randomized optimization algorithm

    Source mlrose documentation
    https://mlrose.readthedocs.io/en/stable/source/tutorial2.html
    """
    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

    # Fitness object
    fitness_coords = mlrose.TravellingSales(coords=coords_list)

    # Optimization problem object
    problem_fit = mlrose.TSPOpt(length=8, fitness_fn=fitness_coords, maximize=False)

    # solve
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state=2)

    print(best_state)
    print(best_fitness)


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
    data = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        test_size=0.2, random_state=1)
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


def kcolor():
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
    fitness = mlrose.MaxKColor(edges)
    problem = mlrose.DiscreteOpt(5,
                                 fitness,
                                 max_val=5,
                                 maximize=True)


def fourpeaks(savedir=None):
    # With early stopping
    t, r = A2.fourpeaks(max_iter=2000, early_stop=10, savedir=savedir)


def main():
    savedir = util.mktmpdir()
    fourpeaks(savedir)


if __name__ == '__main__':
    main()
