import mlrose
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import datajanitor
import numpy as np
import timeit
import util
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def run_mlweight(savedir):
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
    print('\n\n|========= Neural Network =========|\n')
    dataset = datajanitor.getDataset('musk')
    dataset.getData()

    x_train, x_test, y_train, y_test = dataset.partitionData(percent=0.3,
                                                             randomState=10)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 1. Initialize neural network model
    # These hyperparameters are taken from assignment 1 and will not be changed
    ann_hyperparams = {
        'activation': 'relu',
        'hidden_nodes': [200, 200],
        'bias': True,
        'early_stopping': True
    }

    # Set up tunable hyperparameters for each of the optimization algos
    param_grids = [
        {  # SA
            'clf__schedule': [mlrose.GeomDecay(),
                              mlrose.ExpDecay(),
                              mlrose.ArithDecay(),
                              mlrose.ArithDecay(init_temp=1.0, decay=0.01),
                              mlrose.ArithDecay(init_temp=5.0, decay=0.001)],
            'clf__max_attempts': [5000]
        },
        {  # RHC, no actual parameters
            'clf__max_attempts': [5000]
        },
        {  # GA
            'clf__pop_size': [200, 500, 1000, 2000],
            'clf__pop_breed_percent': [0.25, 0.5, 0.75],
            'clf__elite_dreg_ratio': [0.3, 0.6, 0.9],
            'clf__mutation_prob': [0.1, 0.3, 0.6],
            'clf__max_attempts': [5000]
        }
    ]

    opt_algos = ['simulated_annealing', 'random_hill_climb', 'genetic_alg']
    training_times = {}
    training_scores = {}
    scores = []
    for opt, params in zip(opt_algos, param_grids):
        print('Training neural network with {} optimization algorithm'
              .format(opt))
        print('-' * 50)
        nn_model = mlrose.NeuralNetwork(algorithm=opt,
                                        is_classifier=True,
                                        clip_max=5,
                                        random_state=100,
                                        **ann_hyperparams)
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('clf', nn_model)])
        ann = RandomizedSearchCV(estimator=pipe,
                                 n_iter=10,
                                 param_distributions=params,
                                 scoring='balanced_accuracy',
                                 n_jobs=-1,
                                 cv=5)
        ann.fit(x_train, y_train)
        report(ann.cv_results_)

        # Refit with whole dataset
        start = timeit.default_timer()
        final_model = mlrose.NeuralNetwork(algorithm=opt,
                                           **ann_hyperparams)
        # final_model.fit(x_train_scaled, y_train)
        final_pipe = Pipeline([('scaler', StandardScaler()),
                               ('clf', final_model)])
        final_pipe.fit(x_train, y_train)
        train_time = timeit.default_timer() - start
        print('Final model training took {} seconds'.format(train_time))

        # Save best params and training time
        joblib.dump(ann.best_params_, '{}/{}_params.dat'.format(savedir, opt))
        training_times[opt] = train_time
        training_scores[opt] = ann.best_score_

        # Score the model on test
        ypred = final_pipe.predict(x_test_scaled)
        util.confusionMatrix(opt, y_test, ypred,
                             savedir=savedir,
                             scoreType='test')
        scores.append(f1_score(y_test, ypred))
        plt.close('all')
        print('\n')

    # Save final f1 score comparison chart
    util.plotBarScores(scores, opt_algos, '',
                       savedir, phaseName='final')
    f = open('{}/runinfo.txt'.format(savedir), 'w')
    f.write('Train time\n{}\nTrain Scores\n{}'
            .format(training_times, training_scores))


def report(results, n_top=3):
    """
    Report best scores from Grid or RandomizedSearchCV
    :param results:
    :param n_top:
    :return:

    Stolen from SKLearn documentation
    Source:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
