from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
import logging
import numpy as np


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

    proj = trans.fit_transform(X_scaled)
    inv = proj.dot(trans.components_) + X_scaled.mean(axis=0)
    err = ((X_scaled - inv)**2).mean()
    return err
