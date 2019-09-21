import datajanitor
import A1
import util
import argparse
import joblib
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def main():
    args = getArgs()
    datasets = openDatasets(args.datasets)
    classifiers = getClassifiers(args.classifiers)
    runName = datetime.datetime.now().strftime('%m%d%H%M%S')
    baseoutDir = 'output/{}'.format(runName)
    os.makedirs(baseoutDir, exist_ok=True)
    print('Saving output to output/{}'.format(runName))

    for dataset in datasets:
        print('\n\n======= Dataset: {} =======\n'.format(dataset.name))
        outputDir = '{}/{}'.format(baseoutDir, dataset.name)
        os.makedirs(outputDir, exist_ok=True)
        trainX, testX, trainy, testy = dataset.partitionData(percent=0.3, randomState=10)

        # Phase 0: baseline score with defaults
        if 'baseline' in args.phases:
            # Learning curves
            scoreModel(classifiers,
                       trainX,
                       trainy,
                       testX,
                       testy,
                       dataset.scoring,
                       outputDir=outputDir,
                       params=None,
                       dsname=dataset.name)

        # Phase 1: Initial model complexity analysis
        # identify hyperparameters that have some reasonable effect
        if 'mca' in args.phases:
            initialMCA(classifiers=classifiers,
                       X=trainX,
                       y=trainy,
                       scoring=dataset.scoring,
                       savedir=outputDir)

        # Phase 2: grid search over each classifier's relevant parameters
        # these exist per dataset
        fitted = None
        if 'grid' in args.phases:
            # Grid search over parameters that were found to have some effect
            # on overall accuracy. Load from default.
            best_params = util.gridSearch(dataset,
                                          classifiers,
                                          trainX, trainy,
                                          dataset.scoring)

            # Learning curves
            fitted = scoreModel(classifiers,
                                trainX,
                                trainy,
                                testX,
                                testy,
                                dataset.scoring,
                                outputDir=outputDir,
                                params=best_params,
                                scoreType='tuned',
                                dsname=dataset.name)


def scoreModel(classifiers, X, y, testX, testy, scoring,
               outputDir, params, scoreType='baseline',
               dsname=''):
    fitClassifiers = {}
    scores = []
    names = []
    for classifier in classifiers:
        clf, _ = A1.getClfParams(classifier)
        if params is not None:
            # Remove classifier prefix from params
            p = {k.replace('classifier__', ''): v for k, v in params[classifier].items()}
            clf.set_params(**p)

        print('{}: Generating {} learning curve'
              .format(classifier, scoreType))
        print('{}: hyperparameters: '.format(classifier), clf.get_params())
        util.plot_learning_curve(classifier, clf, X,
                                 y, scoring,
                                 savedir=outputDir,
                                 scoreType=scoreType)

        # SVM and ANN need a training epoch graph
        if classifier == 'kernelSVM' or classifier == 'ann':
            util.plotValidationCurve(clf, X, y,
                                     scoring=scoring,
                                     paramName='max_iter',
                                     paramRange=range(100, 2000, 100),
                                     savedir=outputDir,
                                     clfName='{}-{}'.format(classifier, scoreType),
                                     cv=3)

        # To score the model, fit with given parameters and predict
        print('{}: Retraining with best parameters on entire training set'
              .format(classifier))
        pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                   ('classifier', clf)])
        pipeline.fit(X, y)
        ypred = pipeline.predict(testX)
        fitClassifiers[classifier] = pipeline
        scores.append(f1_score(testy, ypred))
        names.append(classifier)

        # Generate confusion matrix
        print('{}: Scoring predictions against test set'
              .format(classifier))
        util.confusionMatrix(classifier, testy, ypred,
                             savedir=outputDir,
                             scoreType=scoreType)

        plt.close('all')

    plotBarScores(scores, names, '', outputDir, phaseName=scoreType)
    plt.close('all')
    return fitClassifiers


def initialMCA(classifiers, X, y, scoring, savedir):
    """
    Run validation curves on every tunable parameter over a small range

    Determine which parameters have effect on the models for the given
    dataset.

    :param classifiers: list of classifier names
    :param X: np.array, features
    :param y: np.array, labels
    :param scoring: string or sklearn scorer function
    :param savedir: directory to save charts or None
    :return:
    """
    tuningParams = {
        # SVM is slow, so very few options, far apart
        'kernelSVM': [['C', [0.75, 1.0]],
                      ['gamma', [0.0001, 0.001]]],
        # Decision tree is fast so do bigger param ranges
        'dt': [['max_depth', range(1, 15, 1)],
               ['min_samples_split', range(2, 10, 2)],
               ['min_samples_leaf', range(2, 10, 2)],
               ['max_features', np.linspace(0.001, 1., 10)],
               ['max_leaf_nodes', range(2, 10)],
               ['class_weight', [{0: 1 / x, 1: 1. - (1. / x)} for x in range(1, 5)]]],
        'ann': [['hidden_layer_sizes', [(20,), (50,), (20, 20), (50, 50), (20, 20, 20)]],
                ['alpha', np.logspace(-5, 0, 5)],
                ['learning_rate_init', [0.1, 0.01, 0.001, 0.0001]]],
        'adaboost': [['n_estimators', range(50, 500, 50)],
                     ['learning_rate', np.logspace(-5, 0, 10)]],
        'knn': [['n_neighbors', range(10, 100, 10)],
                ['p', [1, 2, 3, 4, 5]],
                ['leaf_size', range(20, 50, 10)]]
    }

    for classifier in classifiers:
        valclf, _ = A1.getClfParams(classifier)

        print('-----Model Complexity Analysis: {}-----'
              .format(classifier))

        discreteParams = [valclf]
        discreteNames = ['']
        if classifier == 'kernelSVM':
            linclf, _ = A1.getClfParams(classifier, kernel='linear')
            discreteParams = [valclf, linclf]
            discreteNames = ['rbfKernel', 'linearKernel']
        elif classifier == 'ann':
            tanclf, _ = A1.getClfParams(classifier, activation='tanh')
            discreteParams = [valclf, tanclf]
            discreteNames = ['relu', 'tanh']

        for tuneclf, discreteName in zip(discreteParams, discreteNames):
            for p in tuningParams[classifier]:
                # Fix ANN parameters
                xlabel = p[0]
                xrange = None
                if classifier == 'ann' and p[0] == 'hidden_layer_sizes':
                    xlabel = 'hidden units'
                    xrange = [str(x) for x in p[1]]
                # DT weights are a dict and need to be converted to range
                if classifier == 'dt' and p[0] == 'class_weight':
                    xrange = range(1, 5)
                    xlabel = 'positive class (1) weight'

                clfname = '{}{}'.format(classifier, '-' + discreteName)
                print('{}: Tuning parameter {} in range {}'
                      .format(clfname, p[0], p[1]))
                util.plotValidationCurve(tuneclf, X, y,
                                         scoring=scoring,
                                         paramName=p[0],
                                         paramRange=p[1],
                                         savedir=savedir,
                                         clfName=clfname,
                                         xlabel=xlabel,
                                         xrange=xrange,
                                         cv=3)
                plt.close('all')


def plotBarScores(scores, names, dsname, outputdir, phaseName):
    # barplot code stolen from matplotlib examples
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barh.html
    y_pos = np.arange(len(names))
    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.barh(y_pos, scores, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('F1 Score')
    ax.set_title('Classifier Comparison ({})'
                 .format(dsname))
    # text labeling stolen from stack overflow
    # https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh
    for i, v in enumerate(scores):
        ax.text(0.20, i, '{:.3}'.format(v),
                color='white', va='center', fontweight='bold')

    if outputdir:
        plt.savefig('{}/{}-comp.png'.format(outputdir, phaseName))

    print('F1 scores:')
    for n, s in zip(names, scores):
        print('{}:  {}'.format(n, s))


# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////// Program configuration functions //////////////////
# ////////////////////////////////////////////////////////////////////////////


def openSavedParams(dsname, classifiers):
    filenames = ['models/{}_params.dat'.format('{}_{}'.format(dsname, c))
                 for c in classifiers]
    params = [joblib.load(f) for f in filenames]

    return dict(zip(classifiers, params))


def openDatasets(names):
    if 'all' in names:
        selected = ['adult', 'shoppers', 'news', 'cancer']
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
    validPhases = ['baseline', 'mca', 'grid']
    validData = ['adult', 'shoppers', 'news', 'cancer', 'spam', 'musk']

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
