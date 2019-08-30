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


# Get the datasets and do basic data cleaning
shoppers = datajanitor.getDataset('shoppers')
shoppers.getData(doOHE=True)

# Partition and transform the dataset
shoppersTrainX, shoppersTestX, shoppersTrainY, shoppersTestY = \
    shoppers.partitionData(scale=True)

# Set which classifiers to use
# TODO: get list from command-line
classifiers = ['knn']

# ============ GRID SEARCH ================================================= #

# Train classifier sequentially
for classifier in classifiers:
    print('--- Tuning {} classifier ---'.format(classifier))
    clf, clfParams, scoring = A1.getClfParams(classifier)
    print('Performing grid search over following parameters:\n{}\n'
          .format(clfParams))

    clfCV = GridSearchCV(clf, clfParams, scoring=scoring, cv=5)
    clfCV.fit(shoppersTrainX, shoppersTrainY)

    print('Best parameters found:')
    best = clfCV.best_params_
    print(best)

    print('Performance on test dataset:')
    yPred = clfCV.predict(shoppersTestX)
    print(classification_report(shoppersTestY, yPred))

    print('Generating learning curves')
    clfBest, _, _ = A1.getClfParams(classifier, **best)
    train_sizes, train_scores, valid_scored = learning_curve(
        clfBest, shoppersTrainX, shoppersTrainY, cv=5, scoring=scoring
    )

    util.plot_learning_curve('Learning Curve',
                             train_sizes,
                             train_scores,
                             valid_scored)

# Generate learning curve and ROC AUC

