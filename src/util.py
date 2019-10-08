"""
Utility functions for plotting learner results
"""
import A1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import datetime
import os
import shutil


def mktmpdir(dirname=None):
    """
    Create output directory based on date/time
    :return: string, relative path to new directory
    """
    runName = dirname or datetime.datetime.now().strftime('%m%d%H%M%S')
    baseoutDir = 'output/{}'.format(runName)
    # blast existing directory -- nothing in output is safe!
    shutil.rmtree(baseoutDir, ignore_errors=True)
    os.makedirs(baseoutDir, exist_ok=True)
    print('Saving output to output/{}'.format(runName))
    return baseoutDir


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
        Pipeline([
            ('scale', StandardScaler()),
            ('classifier', clf)
        ]),
        X, y, cv=cv,
        scoring=scoring
    )

    plt.figure()
    plt.title('Learning Curve ({})'.format(title))
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    # plt.ylim(0.4, 1.1)
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
        Pipeline([
            ('scale', StandardScaler()),
            ('classifier', clf)
        ]),
        X, y,
        param_name='classifier__' + paramName,
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
    # plt.ylim(0.4, 1.1)
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
        plt.savefig('{}/{}-vc-{}.png'.format(savedir, clfName, paramName))
    return plt


def gridSearch(dataset, classifiers, X, y, scoring):
    # Train all classifiers sequentially
    bestParams = []
    for classifier in classifiers:
        clf, _ = A1.getClfParams(classifier, dataset.name, pipe=True)
        clfParams = dataset.getGridParams(classifier)
        print('\n\n{}: Performing grid search over following parameters:\n{}\n'
              .format(classifier, clfParams))
        pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                   ('classifier', clf)])
        clfCV = GridSearchCV(pipeline, clfParams, scoring=scoring, cv=5)
        clfCV.fit(X, y)

        print('{}: Best parameters found:'.format(classifier))
        best = clfCV.best_params_
        print(best)
        # save best parameters (not classifier object)
        joblib.dump(best, 'models/{}_{}_params.dat'.format(dataset.name, classifier))
        bestParams.append(best)

    return {k: v for k, v in zip(classifiers, bestParams)}


def scoreClassifier(clfName, ytrue, ypred):
    """
    Standardized scoring metrics. Assumes a fitted classifier.

    :param clfName:
    :param ypred:
    :param ytrue:
    :return:
    """
    print('Scoring performance of {} model:'.format(clfName))
    print(classification_report(ytrue, ypred))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Stolen word-for-word from SK Learn documentation
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    classes = ['negative', 'positive']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def confusionMatrix(clfName, ytrue, ypred,
                    savedir=None,
                    scoreType='score'):
    """
    Create a confusion matrix

    :param clfName: string, name of classifier
    :param ytrue: np array, ground truth labels
    :param ypred: np array, predicted values
    :param savedir: string, path to directory to save figures
    :param scoreType: string, description of scoring round
    :return: None
    """
    title = 'Confusion Matrix ({}: {})'.format(clfName, scoreType)
    plot_confusion_matrix(ytrue, ypred, ['negative', 'positive'],
                          normalize=False,
                          title=title)
    if savedir is not None:
        plt.savefig('{}/{}-{}-cf.png'.format(savedir, clfName, scoreType))


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
