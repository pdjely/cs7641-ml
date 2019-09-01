import datajanitor
import A1
import util

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib


def main():
    # Get the datasets and do basic data cleaning
    shoppers = datajanitor.getDataset('shoppers')
    shoppers.getData(doOHE=True)
    trainX, testX, trainy, testy = shoppers.partitionData(scale=True,
                                                          percent=0.2)

    # Split training set again into train/validation sets
    X, cvX, y, cvy = train_test_split(trainX, trainy,
                                      test_size=0.2,
                                      train_size=0.8,
                                      random_state=1,
                                      stratify=trainy)

    # Set which classifiers to use
    # TODO: get list from command-line
    classifiers = ['dt']

    # Phase 1: Grid search over most paramters to find initial optimal settings
    params = gridSearch(classifiers, X, y, shoppers.scoring)

    # first refit on full training set and then evaluate again validation set
    # This gives initial model analysis for baselines for finer hyperparamter tuning
    for classifier in classifiers:
        clf, _ = A1.getClfParams(classifier)
        clf.set_params(**params[classifier]).fit(X, y)
        scoreClassifier(classifier, clf, cvX, cvy, shoppers.scoring)


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
    print('Scoring performance of {} model:'.format(modelName))
    yPred = clf.predict(X)
    print(classification_report(y, yPred))

    # Here we assume the classifier has already been fit
    print('Generating learning curve')
    train_sizes, train_scores, valid_scored = learning_curve(
        clf, X, y, cv=5, scoring=scoring
    )

    util.plot_learning_curve('Learning Curve',
                             train_sizes,
                             train_scores,
                             valid_scored)
    plt.savefig('models/{}-lc.png'.format(modelName))
    # ROC AUC


if __name__ == '__main__':
    main()
