"""
Utility functions for plotting learner results
"""
import A1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib


def plot_learning_curve(title, clf, X, y, scoring, savedir, cv=5, scoreType=''):

    """
    Generate a simple plot of the test and training learning curve.

    Stolen directly from scikit-learn documentation
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    Parameters
    ----------
    clf : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :param scoreType:
        :param savedir:
        :param X:
        :param y:
        :param scoring:
        :param cv:
        :param clf:
        :param title:
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, scoring=scoring
    )

    plt.figure()
    plt.title('Learning Curve ({})'.format(title))
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    if savedir is not None:
        plt.savefig('{}/{}-{}-lc.png'.format(savedir, title, scoreType))
    return plt


def plotValidationCurve(clf, X, y, scoring, paramName,
                        paramRange, savedir, clfName,
                        xlabel=None, xrange=None, cv=5):
    trainScores, testScores = validation_curve(
        clf,
        X, y,
        param_name=paramName,
        param_range=paramRange,
        cv=cv,
        scoring=scoring
    )
    train_scores_mean = np.mean(trainScores, axis=1)
    train_scores_std = np.std(trainScores, axis=1)
    test_scores_mean = np.mean(testScores, axis=1)
    test_scores_std = np.std(testScores, axis=1)
    paramName = paramName if xlabel is None else xlabel
    paramRange = paramRange if xrange is None else xrange

    plt.figure()
    plt.title('Validation Curve ({})'.format(clfName))
    plt.xlabel(xlabel)
    plt.ylabel('Score')
    plt.ylim(0.6, 1.1)
    lw = 2
    plt.plot(paramRange, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(paramRange, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(paramRange, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(paramRange, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    if savedir is not None:
        plt.savefig('{}/vc_{}_{}.png'.format(savedir, clfName, paramName))
    return plt


def gridSearch(classifiers, X, y, scoring):
    # Train all classifiers sequentially
    print('Dataset uses {} scorer by default'.format(scoring))
    bestParams = []
    for classifier in classifiers:
        print('--- Tuning {} classifier ---'.format(classifier))
        clf, clfParams = A1.getClfParams(classifier)
        print('Performing grid search over following parameters:\n{}\n'
              .format(clfParams))
        clfCV = GridSearchCV(clf, clfParams, scoring=scoring, cv=5)
        clfCV.fit(X, y)

        print('Best parameters found:')
        best = clfCV.best_params_
        print(best)
        # save best parameters (not classifier object)
        joblib.dump(best, 'models/{}_params.dat'.format(classifier))
        bestParams.append(best)

    return {k: v for k, v in zip(classifiers, bestParams)}


def scoreClassifier(modelName, clf, X, y, scoring):
    """
    Standardized scoring metrics. Assumes a fitted classifier.

    :param modelName:
    :param clf:
    :param X:
    :param y:
    :param scoring:
    :return:
    """
    print('Scoring performance of {} model:'.format(modelName))
    yPred = clf.predict(X)
    print(classification_report(y, yPred))


