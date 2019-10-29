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
import logging


def plot_elbow(df):
    ax = df.plot()
    plt.xlabel('n_clusters')
    plt.ylabel('score')
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
    logging.info('TSNE for {} clusters took {} seconds'.format(n_clusters, t1 - t0))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title('tSNE: {} clusters'.format(n_clusters))

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


def cluster(part_range, X, y, savedir, dataset,
            tnse_range=range(3, 5), dr_step=None,
            X_test=None):
    if dr_step:
        km = make_pipeline(StandardScaler(),
                           dr_step,
                           KMeans(random_state=100))
        em = make_pipeline(StandardScaler(),
                           dr_step,
                           GaussianMixture(random_state=100))
        dr_name = '_' + dr_step.__class__.__name__.lower()
        logging.info('Running clustering with {}'.format(dr_name))
    else:
        # KMeans
        km = make_pipeline(StandardScaler(), KMeans(random_state=100))
        # Expectation Maximization
        em = make_pipeline(StandardScaler(), GaussianMixture())
        dr_name = None

    # Get scores for all kmeans estimators
    km_scores = np.zeros((len(part_range), 4))
    km_colnames = None
    km_clusters = []
    km_test_clusters = []
    for i, n in enumerate(part_range):
        km.named_steps['kmeans'].set_params(n_clusters=n)
        km.fit(X, y)
        km_clusters.append(km.predict(X))
        scores, km_colnames = score_clusters(km, X, y)
        km_scores[i] = scores

        if X_test is not None:
            km_test_clusters.append(km.predict(X_test))

        # TNSE relatively expensive to compute, so do for limited range only
        if tnse_range and n in tnse_range:
            km_train_pred = km.predict(X)
            plot_tsne(X, km_train_pred, n)
            plt.savefig('{}/{}-km{}-tsne-{}.png'.format(savedir, dataset, dr_name, n))
    km_scores = pd.DataFrame(km_scores, index=part_range, columns=km_colnames)
    km_scores.to_csv('{}/{}-km{}-clusters.csv'.format(savedir, dataset, dr_name))
    logging.info('KM Scores\n{}'.format(km_scores))
    ax = plot_elbow(km_scores)
    plt.savefig('{}/{}-km{}-elbow.png'.format(savedir, dataset, dr_name))

    # Do same for em estimators
    em_scores = np.zeros((len(part_range), 4))
    em_colnames = None
    em_clusters = []
    em_test_clusters = []
    for i, n in enumerate(part_range):
        em.named_steps['gaussianmixture'].set_params(n_components=n)
        em.fit(X, y)
        em_clusters.append(em.predict(X))
        scores, em_colnames = score_clusters(em, X, y)
        em_scores[i] = scores

        if X_test is not None:
            em_test_clusters.append(em.predict(X_test))

        if tnse_range and n in tnse_range:
            em_train_pred = em.predict(X)
            plot_tsne(X, em_train_pred, n)
            plt.savefig('{}/{}-em{}-tsne-{}.png'.format(savedir, dataset, dr_name, n))
    em_scores = pd.DataFrame(em_scores, index=part_range, columns=em_colnames)
    em_scores.to_csv('{}/{}-em{}-clusters.csv'.format(savedir, dataset, dr_name))
    logging.info('EM Scores\n{}'.format(em_scores))
    plot_elbow(em_scores)
    plt.savefig('{}/{}-em{}-elbow.png'.format(savedir, dataset, dr_name))

    plt.close('all')

    if X_test is None:
        return km_clusters, em_clusters
    else:
        return km_clusters, em_clusters, km_test_clusters, em_test_clusters
