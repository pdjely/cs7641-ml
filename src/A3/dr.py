from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
import logging
import numpy as np
import pandas as pd


def pca(X, y, n_components=0.95):
    pipe = make_pipeline(StandardScaler(), PCA(n_components=n_components))
    reduced = pipe.fit_transform(X)
    logging.info('pca found {} components'.format(pipe.named_steps['pca'].n_components_))
    plt.scatter(reduced[y == 0, 0], reduced[y == 0, 1])
    plt.scatter(reduced[y == 1, 0], reduced[y == 1, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    return pipe


def ica(X, y, n_components=None):
    pipe = make_pipeline(StandardScaler(),
                         FastICA(n_components=n_components))
    reduced = pipe.fit_transform(X)
    logging.info('ICA component shape: {}'.format(pipe.named_steps['fastica'].components_.shape))
    # logging.info(pipe.named_steps['fastica'].components_)

    return pipe


def rfselect(X, y):
    """
    Use recursive feature selection with random forest to select optimal feat

    Source:
        https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV
    :param X:
    :param y:
    :return:
    """
    pipe = make_pipeline(StandardScaler(),
                         RFECV(RandomForestClassifier(n_estimators=500),
                               step=5,
                               scoring='f1_weighted',
                               cv=3,
                               n_jobs=-1))
    pipe.fit(X, y)
    logging.info('Random Forest: Optimal number of features: {}'
                 .format(pipe.named_steps['rfecv'].n_features_))

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(pipe.named_steps['rfecv'].grid_scores_) + 1),
             pipe.named_steps['rfecv'].grid_scores_)

    # Don't want to re-do recursive cv at run time
    rf = Pipeline([('scaler', StandardScaler()),
                   ('randomforest', SelectFromModel(RandomForestClassifier(n_estimators=500),
                                                    threshold=-np.inf,
                                                    max_features=pipe.named_steps['rfecv'].n_features_))])
    return rf


def random_projection(X, y, n_components='auto'):
    # Stolen from sklearn docs
    # Source:
    #   https://scikit-learn.org/stable/modules/random_projection.html
    pipe = make_pipeline(StandardScaler(),
                         GaussianRandomProjection(n_components))
    X_new = pipe.fit_transform(X)
    logging.info('RP transformed input to {}'.format(X_new.shape))

    return pipe


def recon_error(trans, X):
    """
    Compute reconstruction error
    :param trans: transformer
    :param X: training samples
    :return: reconstruction error

    Source:
        https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    proj = trans.transform(X_scaled)
    inv = proj.dot(trans.components_) + X_scaled.mean(axis=0)
    err = ((X_scaled - inv)**2).mean()

    return err


def avg_kurtosis(projections):
    X = pd.DataFrame(projections)
    return np.mean(X.kurtosis(axis=0))
