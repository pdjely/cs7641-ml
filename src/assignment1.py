import datajanitor
import A1
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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

# Fit an SVM
# Adapted from sklearn example:
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html#sphx-glr-auto-examples-svm-plot-svm-scale-c-py
svm, svmParams = A1.getClfParams('kernelSVM')

print('Tuning hyperparameters')
# TODO: should parameterize scoring, n_jobs, and cv number
svmCV = GridSearchCV(svm, svmParams, scoring='f1_weighted', n_jobs=None, cv=5)
svmCV.fit(shoppersTrainX, shoppersTrainY)

print("Best parameters set found on development set:")
print(svmCV.best_params_)

yPred = svmCV.predict(shoppersTestX)
print(classification_report(shoppersTestY, yPred))

# Generate learning curve and ROC AUC
