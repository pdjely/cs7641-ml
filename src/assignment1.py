import datajanitor
import A1
import util
from sklearn.model_selection import train_test_split
import argparse
import joblib
import datetime
import os
import numpy as np


def main():
    args = getArgs()
    datasets = openDatasets(args.datasets)
    classifiers = getClassifiers(args.classifiers)
    runName = datetime.datetime.now().strftime('%m%d%H%M%S')
    baseoutDir = 'output/{}'.format(runName)
    os.makedirs(baseoutDir, exist_ok=True)
    print('Saving output to output/{}'.format(runName))

    # Do an 80/20 split train/test. Training set will then be split further
    # (50/50) into a grid search training set and validation set for
    # model complexity analysis
    for dataset in datasets:
        print('===== Dataset: {} ====='.format(dataset.name))
        outputDir = '{}/{}'.format(baseoutDir, dataset.name)
        os.makedirs(outputDir, exist_ok=True)
        # This is not optimal since we're scaling our validation set also
        # but letting it slide to reduce some book keeping
        trainX, testX, trainy, testy = dataset.partitionData(scale=True,
                                                             percent=0.2)

        # If doing initial grid search, divide training 50/50 into train/validation
        # otherwise just set 20% of total aside for model cross validation
        splitVal = 0.5 if 'grid' in args.phases else 0.25
        gridX, valX, gridy, valy = train_test_split(trainX, trainy,
                                                    test_size=splitVal,
                                                    random_state=1,
                                                    shuffle=True,
                                                    stratify=trainy)

        # Phase 1: Grid search over most parameters to find initial optimal settings
        # these are considered 'universal' (and time-consuming to run), so these
        # are saved in a standard directory for later use
        if 'grid' in args.phases:
            params = util.gridSearch(classifiers, gridX, gridy, dataset.scoring)
        else:
            print('Skipping gridsearch. Loading parameters from /models...')
            params = openSavedParams(classifiers)

        # Generate baseline scores on the test set for found initial parameters
        # from grid search and get learning curves.
        if 'baseline' in args.phases:
            scoreModel(classifiers, params, trainX, trainy,
                       testX, testy, dataset.scoring, outputDir)

        # Phase 2: Model complexity analysis--tune two hyperparameters
        # and show validation curves
        if 'mca' in args.phases:
            tuneModel(classifiers=classifiers,
                      params=params,
                      X=gridX,
                      y=gridy,
                      scoring=dataset.scoring,
                      savedir=outputDir)


def scoreModel(classifiers, params, X, y, testX, testy, scoring,
               outputDir, scoreType='baseline'):
    for classifier in classifiers:
        clf, _ = A1.getClfParams(classifier)
        clf.set_params(**params[classifier])

        print('Generating learning curve')
        util.plot_learning_curve(classifier, clf, X,
                                 y, scoring,
                                 savedir=outputDir,
                                 scoreType=scoreType)

        print('Scoring initial parameters against test set')
        clf.fit(X, y)
        util.scoreClassifier(classifier, clf, testX,
                             testy, scoring=scoring)


def tuneModel(classifiers, params, X, y, scoring, savedir):
    tuningParams = {
        'kernelSVM': [['C', np.linspace(0.001, 10, 10)],
                      ['gamma', np.logspace(-5, 0, 10)]],
        'dt': [['max_depth', range(1, 10, 1)],
               ['min_samples_split', range(2, 10, 2)],
               ['min_samples_leaf', range(2, 10, 2)],
               ['max_features', np.linspace(0.001, 1., 10)],
               ['max_leaf_nodes', range(2, 10)],
               ['class_weight', [{0: 1 / x, 1: 1. - (1. / x)} for x in range(1, 5)]]],
        'ann': [['hidden_layer_sizes', [(x,) for x in range(25, 105, 25)]],
                ['hidden_layer_sizes', [(50,), (50, 50), (50, 50, 50)]],
                ['alpha', np.logspace(-5, 0, 5)],
                ['beta_1', [0.5, 0.75, 0.9]],
                ['learning_rate_init', [0.0001, 0.001, 0.01, 0.1]]],
        'adaboost': [['n_estimators', range(50, 500, 50)],
                     ['learning_rate', np.logspace(-5, 0, 10)]],
        'knn': [['n_neighbors', range(10, 100, 10)],
                ['p', [1, 2, 3, 4, 5]],
                ['leaf_size', range(20, 50, 10)]]
    }

    for classifier in classifiers:
        valclf, _ = A1.getClfParams(classifier)
        if 'ann' in params.keys():
            params['ann']['activation'] = 'relu'
            params['ann']['solver'] = 'adam'
        # valclf.set_params(**params[classifier])

        print('-----Model Complexity Analysis-----')
        print('Using set parameters:\n{}'.format(params))

        for p in tuningParams[classifier]:
            # Fix ANN parameters
            xlabel = p[0]
            xrange = None
            if classifier == 'ann' and p[0] == 'hidden_layer_sizes':
                xlabel = 'hidden units' if len(p[1][1]) == 1 else 'hidden layers'
                if xlabel == 'hidden units':
                    xrange = [x[0] for x in p[1]]
                else:
                    xrange = [1, 2, 3]    # hard code 3 layers
            # DT weights are a dict and need to be converted to range
            if classifier == 'dt' and p[0] == 'class_weight':
                xrange = range(1, 5)
                xlabel = 'positive class (1) weight'

            print('{}: Tuning parameter {} in range {}'
                  .format(classifier, p[0], p[1]))
            plt = util.plotValidationCurve(valclf, X, y,
                                           scoring=scoring,
                                           paramName=p[0],
                                           paramRange=p[1],
                                           savedir=savedir,
                                           clfName=classifier,
                                           xlabel=xlabel,
                                           xrange=xrange,
                                           cv=3)
            plt.clf()


def finalScore():
    pass

# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////// Program configuration functions //////////////////
# ////////////////////////////////////////////////////////////////////////////


def openSavedParams(classifiers):
    filenames = ['models/{}_params.dat'.format(c) for c in classifiers]
    params = [joblib.load(f) for f in filenames]

    return dict(zip(classifiers, params))


def openDatasets(names):
    if 'all' in names:
        selected = ['adult', 'shoppers']
    else:
        selected = names
    datasets = [datajanitor.getDataset(s) for s in selected]

    for i, _ in enumerate(datasets):
        datasets[i].getData(doOHE=True)

    return datasets


def getClassifiers(names):
    if 'all' in names:
        return ['dt', 'ann', 'knn', 'kernelSVM', 'adaboost']

    classifiers = []
    for n in names:
        if n == 'svm':
            classifiers.append('kernelSVM')
        elif n == 'boost':
            classifiers.append('adaboost')
        else:
            classifiers.append(n)

    return classifiers


def getArgs():
    parser = argparse.ArgumentParser(description='CS7641 Assignment 1')

    validClassifiers = ['dt', 'ann', 'svm', 'boost', 'knn']
    validPhases = ['grid', 'mca']
    validData = ['adult', 'shoppers']

    parser.add_argument('-c', '--classifiers',
                        help='Space-separated list of classifiers (default: all)',
                        choices=validClassifiers, default=validClassifiers,
                        nargs='+')
    parser.add_argument('-d', '--datasets',
                        help='Space-separated list of datasets (default: all)',
                        choices=validData, default=validData,
                        nargs='+')
    parser.add_argument('-p', '--phases',
                        help='Analytical phases to run (default: all)',
                        choices=validPhases, default=validPhases,
                        nargs='+')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
