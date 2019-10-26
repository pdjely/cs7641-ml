import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
from timeit import default_timer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


def plot_elbow(df):
    ax = df.plot()
    plt.show()


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
    print('TSNE took {} seconds'.format(t1 - t0))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_clusters):
        c = clusters == i
        ax.scatter(y[c, 0], y[c, 1])
    plt.show()


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
