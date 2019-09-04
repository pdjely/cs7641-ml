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
        trainX, testX, trainy, testy = dataset.partitionData(scale=True,
                                                             percent=0.2)

        # If doing initial grid search, divide training 50/50 into train/validation
        # otherwise just set 20% of total aside for model cross validation
        splitVal = 0.5 if 'grid' in args.phases else 0.25
        gridX, valX, gridy, valy = train_test_split(trainX, trainy,
                                                    test_size=splitVal,
                                                    random_state=1,
                                                    stratify=trainy)

        # Phase 1: Grid search over most parameters to find initial optimal settings
        # these are considered 'universal' (and time-consuming to run), so these
        # are saved in a standard directory for later use
        if 'grid' in args.phases:
            params = util.gridSearch(classifiers, gridX, gridy, dataset.scoring)
        else:
            print('Skipping gridsearch. Loading parameters from /models...')
            params = openSavedParams(classifiers)

        # Generate scores for found initial parameters from grid search and
        # get learning curves. Then score against test set.
        # TODO: refactor AND add 'baseline' to output filename
        for classifier in classifiers:
            clf, _ = A1.getClfParams(classifier)
            clf.set_params(**params[classifier])

            print('Generating learning curve')
            util.plot_learning_curve(classifier, clf, gridX,
                                     gridy, dataset.scoring,
                                     savedir=outputDir)

            print('Scoring initial parameters against test set')
            clf.fit(gridX, gridy)
            util.scoreClassifier(classifier, clf, testX,
                                 testy, scoring=dataset.scoring)

        # Phase 2: Model complexity analysis--tune two hyperparameters
        # and show validation curves
        if 'mca' in args.phases:
            tuneModel(classifiers=classifiers,
                      params=params,
                      X=valX,
                      y=valy,
                      scoring=dataset.scoring,
                      savedir=outputDir)


def tuneModel(classifiers, params, X, y, scoring, savedir):
    tuningParams = {
        'kernelSVM': [['C', np.linspace(0, 10, 10)]],
        'dt': [['max_depth', range(2, 10, 1)]],
        'ann': [['hidden_layer_sizes', [(x,) for x in range(50, 200, 25)]],
                ['alpha', np.logspace(-5, 0, 10)]],
        'adaboost': [['n_estimators', range(30, 100, 10)],
                     ['learning_rate', np.logspace(-5, 0, 10)]],
        'knn': [['n_neighbors', range(10, 100, 10)],
                ['leaf_size', range(20, 50, 10)]]
    }

    for classifier in classifiers:
        valclf, _ = A1.getClfParams(classifier)
        valclf.set_params(**params[classifier])

        for p in tuningParams[classifier]:
            util.plotValidationCurve(valclf, X, y,
                                     scoring=scoring,
                                     paramName=p[0],
                                     paramRange=p[1],
                                     savedir=savedir,
                                     clfName=classifier)

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

    return names


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
