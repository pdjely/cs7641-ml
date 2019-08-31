import datajanitor
import A1
import util

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib


def gridSearch(classifiers, dataset):
    X, testX, y, testy = dataset.partitionData(scale=True)
    # Train all classifiers sequentially
    print('Dataset uses {} scorer by default'.format(dataset.scoring))
    bestParams = []
    for classifier in classifiers:
        print('--- Tuning {} classifier ---'.format(classifier))
        clf, clfParams = A1.getClfParams(classifier)
        scoring = shoppers.getScorer()
        print('Performing grid search over following parameters:\n{}\n'
              .format(clfParams))

        clfCV = GridSearchCV(clf, clfParams, scoring=scoring, cv=5)
        clfCV.fit(X, y)

        print('Best parameters found:')
        best = clfCV.best_params_
        print(best)
        # save best parameters (not classifier object)
        joblib.dump(best, 'models/{}.dat'.format(classifier))
        bestParams.append(best)

        print('Performance on test dataset:')
        yPred = clfCV.predict(testX)
        print(classification_report(testy, yPred))

        print('Generating learning curves')
        clfBest, _ = A1.getClfParams(classifier, **best)
        train_sizes, train_scores, valid_scored = learning_curve(
            clfBest, X, y, cv=5, scoring=scoring
        )

        util.plot_learning_curve('Learning Curve',
                                 train_sizes,
                                 train_scores,
                                 valid_scored)

        # ROC AUC
        return {k: v for k, v in zip(classifiers, bestParams)}


if __name__ == '__main__':
    # Get the datasets and do basic data cleaning
    shoppers = datajanitor.getDataset('shoppers')
    shoppers.getData(doOHE=True)

    # Set which classifiers to use
    # TODO: get list from command-line
    classifiers = ['dt']

    print(gridSearch(classifiers, shoppers))

