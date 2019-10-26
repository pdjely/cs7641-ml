import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import datajanitor
import util
import A3

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline, Pipeline
import sklearn.metrics as metrics


def cluster(X, y):
    part_range = range(2, 21)
    # KMeans
    km = [make_pipeline(StandardScaler(), KMeans(n_clusters=i, random_state=100))
          for i in part_range]
    # Expectation Maximization
    em = [make_pipeline(StandardScaler(), GaussianMixture(n_components=i))
          for i in part_range]

    # Get scores for all kmeans estimators
    km_scores = np.zeros((len(part_range), 4))
    km_colnames = None
    for i, est in enumerate(km):
        km[i].fit(X)
        scores, km_colnames = A3.score_clusters(km[i], X, y)
        km_scores[i] = scores
    km_scores = pd.DataFrame(km_scores, index=part_range, columns=km_colnames)
    print('KM Scores\n', km_scores)
    # A3.plot_elbow(km_scores)

    # Do same for em estimators
    em_scores = np.zeros((len(part_range), 4))
    em_colnames = None
    for i, est in enumerate(km):
        em[i].fit(X)
        scores, em_colnames = A3.score_clusters(em[i], X, y)
        em_scores[i] = scores
    em_scores = pd.DataFrame(em_scores, index=part_range, columns=em_colnames)
    print('EM Scores\n', em_scores)
    # A3.plot_elbow(em_scores)

    # Elbow says 3 or 4 clusters best with dataset
    # TNSE plot shows three distinct clusters
    km_train_pred = km[1].predict(X)
    em_train_pred = em[1].predict(X)
    A3.plot_tsne(X, km_train_pred, 3)
    A3.plot_tsne(X, em_train_pred, 3)

    return km[1], em[1]


def main():
    for ds in ['musk']:
        dataset = datajanitor.getDataset(ds)
        dataset.getData()
        x_train, x_test, y_train, y_test = dataset.partitionData(percent=0.3, randomState=10)

        # ********************* #
        # **** Clustering  **** #
        # ********************* #
        km, em = cluster(x_train, y_train)

        # ************************ #
        # **** Dim Reduction  **** #
        # ************************ #

        # PCA
        pca = make_pipeline(StandardScaler(), PCA(0.95))
        pca.fit_transform(x_train)
        print('pca found', pca.named_steps['pca'].n_components_, ' components')
        print(pca.named_steps['pca'].components_)

        # ICA
        ica = make_pipeline(StandardScaler(), FastICA(max_iter=500))
        ica.fit_transform(x_train)
        print('Component shape:', ica.named_steps['fastica'].components_.shape)
        print(ica.named_steps['fastica'].components_)


if __name__ == '__main__':
    main()
