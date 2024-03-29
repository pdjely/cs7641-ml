import numpy as np
import pandas as pd
import argparse
import datajanitor
import util
import A3
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import logging
from statistics import mode, StatisticsError


def main():
    args = get_args()
    savedir = util.mktmpdir(args.outdir)

    # Logging copy-pasted from logging cookbook
    # http://docs.python.org/howto/logging-cookbook.html#logging-to-multiple-destinations
    logging.basicConfig(format='%(asctime)s %(message)s',
                        filename='{}/output.log'.format(savedir),
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    for ds in ['musk', 'shoppers']:
        formatter = logging.Formatter('{}: %(levelname)-8s %(message)s'
                                      .format(ds))
        console.setFormatter(formatter)
        logging.info('==========Starting {} Dataset =============='
                     .format(ds))

        dataset = datajanitor.getDataset(ds)
        dataset.getData()
        x_train, x_test, y_train, y_test = \
            dataset.partitionData(percent=0.3, randomState=10)

        # ********************* #
        # **** Clustering  **** #
        # ********************* #
        if 'cluster' in args.phase or 'cluster-ann' in args.phase:
            km_train_clust, em_train_clust, km_test_clust, em_test_clust = \
                A3.cluster(range(2, 21), x_train, y_train, savedir, ds,
                           tnse_range=None,
                           X_test=x_test)

            # ******************************* #
            # **** Clusters as features  **** #
            # ******************************* #
            # one-hot encode and then add clusters to train and test features
            cluster_nn_scores = {
                'km': [],
                'em': [],
                'km+em': [],
                'km_only': [],
                'em_only': [],
                'kmem_only': []
            }
            if 'cluster-ann' in args.phase:
                for i in range(5):
                    km_x_train = add_cluster_dims(x_train, km_train_clust[i])
                    km_x_test = add_cluster_dims(x_test, km_test_clust[i])
                    em_x_train = add_cluster_dims(x_train, em_train_clust[i])
                    em_x_test = add_cluster_dims(x_test, em_test_clust[i])
                    km_score = cluster_nn(km_x_train, y_train, km_x_test, y_test,
                                          savedir, ds, 'km{}'.format(i+2))
                    em_score = cluster_nn(em_x_train, y_train, em_x_test, y_test,
                                          savedir, ds, 'em{}'.format(i+2))
                    kmem_x_train = add_cluster_dims(km_x_train, em_train_clust[i])
                    kmem_x_test = add_cluster_dims(km_x_test, em_test_clust[i])
                    kmem_score = cluster_nn(kmem_x_train, y_train, kmem_x_test, y_test,
                                            savedir, ds, 'kmem{}'.format(i+2))
                    # do only clusters
                    km_only = cluster_nn(km_train_clust[i].reshape(-1, 1), y_train, km_test_clust[i].reshape(-1, 1),
                                         y_test, savedir, ds, 'km_only{}'.format(i+2))
                    em_only = cluster_nn(km_train_clust[i].reshape(-1, 1), y_train, km_test_clust[i].reshape(-1, 1),
                                         y_test, savedir, ds, 'em_only{}'.format(i + 2))
                    kmem_only = cluster_nn(np.append(km_train_clust[i].reshape(-1, 1),
                                                     em_train_clust[i].reshape(-1, 1), axis=1),
                                           y_train,
                                           np.append(km_test_clust[i].reshape(-1, 1),
                                                     em_test_clust[i].reshape(-1, 1), axis=1),
                                           y_test, savedir, ds, 'kmem_only{}'.format(i + 2))
                    util.plotBarScores([km_score, em_score, kmem_score, km_only, em_only, kmem_only],
                                       ['km-ann', 'em-ann', 'kmem-ann', 'km_only', 'em_only', 'kmem_only'],
                                       ds, savedir,
                                       phaseName='{}-cluster-{}-ann'.format(ds, i+2))
                    cluster_nn_scores['km'].append(km_score)
                    cluster_nn_scores['em'].append(em_score)
                    cluster_nn_scores['km+em'].append(kmem_score)
                    cluster_nn_scores['km_only'].append(km_only)
                    cluster_nn_scores['em_only'].append(em_only)
                    cluster_nn_scores['kmem_only'].append(kmem_only)
                    plt.close('all')
                pd.DataFrame.from_dict(data=cluster_nn_scores).to_csv('{}/{}-clusternn.csv'
                                                                      .format(savedir, ds))

        # ************************ #
        # **** Dim Reduction  **** #
        # ************************ #
        # You actually have to do dimension reduction, there is no choice
        dr_steps = dr(x_train, y_train, savedir, ds)

        # *********************** #
        # **** DR + Cluster  **** #
        # *********************** #
        if 'dr-cluster' in args.phase:
            km_train_clust, em_train_clust, km_test_clust, em_test_clust = \
                dr_cluster(x_train, y_train, x_test, dr_steps, savedir, ds)

        # ******************* #
        # **** DR + ANN  **** #
        # ******************* #
        dr_ann(x_train, y_train, x_test, y_test,
               dr_steps, savedir, ds)


def dr(X, y, savedir, ds):
    # First do pca
    pca_pipe = A3.pca(X, y)
    pca = pca_pipe.named_steps['pca']
    plt.savefig('{}/{}-pca.png'.format(savedir, ds))
    np.savetxt('{}/{}-pca-ev.csv'.format(savedir, ds),
               pca.explained_variance_)
    np.savetxt('{}/{}-pca-ev-ratio.csv'.format(savedir, ds),
               pca.explained_variance_ratio_)
    plt.close('all')
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('n_components')
    plt.ylabel('explained variance (%)')
    plt.savefig('{}/{}-pca-ev.png'.format(savedir, ds))
    plt.close('all')
    reconstruction_error = A3.recon_error(pca, X)
    logging.info('PCA reconstruction error: {}'.format(reconstruction_error))

    # second ICA
    ica = None
    max_kurtosis = -np.inf
    ica_range = range(10, X.shape[1], 10)
    kurt_per_comp = []
    for i in ica_range:
        ica_pipe = A3.ica(X, y, i)
        # This should be selected by kurtosis
        kurt = A3.avg_kurtosis(ica_pipe.transform(X))
        kurt_per_comp.append(kurt)
        logging.info('ICA {} average kurtosis: {}'.format(i, kurt))
        if kurt > max_kurtosis:
            ica = ica_pipe.named_steps['fastica']
            max_kurtosis = kurt
    logging.info('ICA max kurtosis {} with {} components'
                 .format(max_kurtosis, ica.components_.shape[0]))
    plt.plot(ica_range, kurt_per_comp)
    plt.xlabel('n_components')
    plt.ylabel('mean kurtosis')
    plt.savefig('{}/{}-ica-kurtosis.png'.format(savedir, ds))

    # RP
    logging.info('Starting randomized projection...')
    rp_errors = []
    rp = None
    best_rp_err = np.inf
    reconstruction_errors = []
    for rp_run in range(10):
        logging.info('RP iteration {}'.format(rp_run))
        best_run = np.inf
        run_errors = []
        for i in range(10, X.shape[1], 10):
            rp_pipe = A3.random_projection(X, y, i)
            err = A3.recon_error(rp_pipe.named_steps['gaussianrandomprojection'], X)
            run_errors.append(err)
            logging.info('RP {} components reconstruction error: {}'
                         .format(i, err))
            if err < best_rp_err:
                rp = rp_pipe.named_steps['gaussianrandomprojection']
                best_rp_err = err
            if err < best_run:
                best_run = err
        reconstruction_errors.append(run_errors)
        rp_errors.append(best_run)
    pd.DataFrame(reconstruction_errors, columns=range(10, X.shape[1], 10)).to_csv('{}/{}-rp-reconstruction.csv'
                                                                                  .format(savedir, ds))
    # Manually set random projection
    if ds == 'musk':
        rp.set_params(n_components=50)
    elif ds == 'shoppers':
        rp.set_params(n_components=30)

    plt.figure()
    plt.plot(range(10), rp_errors)
    plt.xlabel('iteration')
    plt.ylabel('reconstruction error')
    plt.savefig('{}/{}-rp-reconstruction.png'.format(savedir, ds))
    plt.close('all')
    logging.info('RP best n_components: {}'.format(rp.n_components_))

    # TODO: fourth dimension reduction
    rf_pipe = A3.rfselect(X, y)
    rf = rf_pipe.named_steps['randomforest']

    return [pca, ica, rp, rf]


def add_cluster_dims(X, clusters):
    oh = OneHotEncoder(categories='auto')
    c = oh.fit_transform(clusters.reshape(-1, 1)).toarray()
    return np.append(X, c, axis=1)


def dr_cluster(X, y, X_test, dr_steps, savedir, ds):
    """
    Apply dimensionality reduction and then KMeans and EM clustering
    :param X: np array, training samples
    :param y: np array, labels
    :param X_test: np.array, test data
    :param dr_steps: list of dimensionality reduction objects
    :param savedir: string, output directory
    :param ds: string, name of dataset
    :return: tuple, best clusters for each dr type
    """
    cluster_idx = {
        'musk': 1,
        'cancer': 0,
        'shoppers': 0
    }
    best_km = []
    best_em = []
    best_test_km = []
    best_test_em = []

    for dr_step in dr_steps:
        km, em, km_test, em_test = A3.cluster(range(2, 21), X, y,
                                              savedir, ds,
                                              tnse_range=range(3, 5),
                                              dr_step=dr_step,
                                              X_test=X_test)

        best_km.append(km[cluster_idx[ds]])
        best_em.append(em[cluster_idx[ds]])
        best_test_km.append(km_test[cluster_idx[ds]])
        best_test_em.append(em_test[cluster_idx[ds]])

    return best_km, best_em, best_test_km, best_test_em


def cluster_nn(X_train, y_train, X_test, y_test, savedir, ds, cluster_type):
    logging.info('Running neural net with {} clusters as features'
                 .format(cluster_type))
    pipe = A3.baseline_ann(X_train, y_train, ds)
    ypred = pipe.predict(X_test)
    util.confusionMatrix('{}-{}-ann'.format(ds, cluster_type),
                         y_test, ypred, savedir)
    return f1_score(y_test, ypred)


def dr_ann(X_train, y_train, X_test, y_test, dr_steps, savedir, ds,
           cluster=None):
    if cluster is not None:
        c = ' with clustering from {}'.format(cluster)
    else:
        c = ''
    logging.info('ANN: Running baseline neural net' + c)
    baseline = A3.baseline_ann(X_train, y_train, ds)
    ypred = baseline.predict(X_test)
    util.confusionMatrix('{}{}-baseline'.format(ds, cluster),
                         y_test, ypred, savedir)

    scores = [f1_score(y_test, ypred)]
    score_names = ['baseline']
    for dr_step in dr_steps:
        drname = dr_step.__class__.__name__.lower()
        score_names.append(drname)
        logging.info('ANN: Running neural net with {} dimension reduction'
                     .format(drname) + c)
        # Get trained ann with dr
        ann = A3.dr_ann(X_train, y_train, dr_step, ds)
        ypred = ann.predict(X_test)
        util.confusionMatrix('{}-{}{}'.format(ds, drname, cluster),
                             y_test, ypred, savedir)
        scores.append(f1_score(y_test, ypred))

    logging.info('ANN {} F1 Scores: {}'.format(c, scores))
    util.plotBarScores(scores, score_names, ds, savedir,
                       phaseName='{}-{}'.format(ds, cluster))
    plt.close('all')


def get_args():
    parser = argparse.ArgumentParser(description='CS7641 Assignment 3')

    phases = ['cluster', 'dr', 'dr-cluster', 'dr-nn', 'cluster-ann']
    validData = ['musk', 'shoppers']
    parser.add_argument('-p', '--phase',
                        help='Space-separated list of phases to run (default: all)',
                        choices=phases, default=phases,
                        nargs='+')
    parser.add_argument('-d', '--datasets',
                        help='Space-separated list of datasets (default: all)',
                        choices=validData, default=validData,
                        nargs='+')
    parser.add_argument('-o', '--outdir',
                        help='Directory to save files to (optional: default timestamp)',
                        nargs='?', default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
