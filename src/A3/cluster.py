import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
from timeit import default_timer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import pandas as pd


def plot_elbow(df):
    ax = df.plot()
    return ax


def plot_tsne(X, clusters, n_clusters):
    # Stolen from sklearn documentation
    # Source:
    #   https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
    tsne = make_pipeline(StandardScaler(),
                         TSNE(n_components=2, init='random', random_state=100,
                              perplexity=30.0))
    t0 = default_timer()
    y = tsne.fit_transform(X)
    t1 = default_timer()
    print('TSNE for {} clusters took {} seconds'.format(n_clusters, t1 - t0))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_clusters):
        c = clusters == i
        ax.scatter(y[c, 0], y[c, 1])


def score_clusters(estimator, X, y):
    # Elbow graph stolen from blog
    # Source:
    #   https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
    ypred = estimator.predict(X)

    # internal cluster scores -- no ground truth
    sil_score = metrics.silhouette_score(X, ypred, metric='euclidean')
    db_score = metrics.davies_bouldin_score(X, ypred)

    # external cluster scores -- based on labels
    homogeneity, completeness, vmeasure = \
        metrics.homogeneity_completeness_v_measure(y, ypred)
    ami = metrics.adjusted_mutual_info_score(y, ypred, average_method='arithmetic')

    score_names = ['silhouette', 'davies-bouldin', 'v-measure', 'adj mutual inf']
    return np.array([sil_score, vmeasure, ami, db_score]), score_names


def cluster(part_range, X, y, savedir, dataset):
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
        scores, km_colnames = score_clusters(km[i], X, y)
        km_scores[i] = scores
    km_scores = pd.DataFrame(km_scores, index=part_range, columns=km_colnames)
    km_scores.to_csv('{}/{}-km-clusters.csv'.format(savedir, dataset))
    print('KM Scores\n', km_scores)
    ax = plot_elbow(km_scores)
    plt.savefig('{}/{}-km-elbow.png'.format(savedir, dataset))

    # Do same for em estimators
    em_scores = np.zeros((len(part_range), 4))
    em_colnames = None
    for i, est in enumerate(km):
        em[i].fit(X)
        scores, em_colnames = score_clusters(em[i], X, y)
        em_scores[i] = scores
    em_scores = pd.DataFrame(em_scores, index=part_range, columns=em_colnames)
    em_scores.to_csv('{}/{}-em-clusters.csv'.format(savedir, dataset))
    print('EM Scores\n', em_scores)
    plot_elbow(em_scores)
    plt.savefig('{}/{}-em-elbow.png'.format(savedir, dataset))

    # Elbow says 3 or 4 clusters best with dataset
    # TNSE plot shows three distinct clusters
    for i in range(3, 5):
        km_train_pred = km[i].predict(X)
        em_train_pred = em[i].predict(X)
        plot_tsne(X, km_train_pred, i)
        plt.savefig('{}/{}-km-tsne-{}.png'.format(savedir, dataset, i))
        plot_tsne(X, em_train_pred, i)
        plt.savefig('{}/{}-em-tsne-{}.png'.format(savedir, dataset, i))

    plt.close('all')
    return km, em
