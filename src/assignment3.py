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


def main():
    args = get_args()
    savedir = util.mktmpdir(args.outdir)

    for ds in ['musk', 'shoppers']:
        # Logging copy-pasted from logging cookbook
        # http://docs.python.org/howto/logging-cookbook.html#logging-to-multiple-destinations
        logging.basicConfig(format='%(asctime)s %(message)s',
                            filename='{}/{}.log'.format(savedir, ds),
                            level=logging.INFO)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('{}: %(levelname)-8s %(message)s'
                                      .format(ds))
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        dataset = datajanitor.getDataset(ds)
        dataset.getData()
        x_train, x_test, y_train, y_test = \
            dataset.partitionData(percent=0.3, randomState=10)

        # ********************* #
        # **** Clustering  **** #
        # ********************* #
        if 'cluster' in args.phase:
            A3.cluster(range(2, 21), x_train, y_train, savedir, ds)

        # ************************ #
        # **** Dim Reduction  **** #
        # ************************ #
        # You actually have to do dimension reduction, there is no choice
        dr_steps = dr(x_train, y_train, savedir, ds)

        # *********************** #
        # **** DR + Cluster  **** #
        # *********************** #
        if 'dr-cluster' in args.phase or 'dr-cluster-nn' in args.phase:
            km_test_clust, em_test_clust = dr_cluster(x_test,
                                                      y_test,
                                                      dr_steps,
                                                      savedir,
                                                      ds)
            km_train_clust, em_train_clust = dr_cluster(x_train,
                                                        y_train,
                                                        dr_steps,
                                                        savedir,
                                                        ds)

            # ******************************* #
            # **** Clusters as features  **** #
            # ******************************* #
            # one-hot encode and then add clusters to train and test features
            if 'dr-cluster-nn' in args.phase:
                for i, dr_step in enumerate(dr_steps):
                    km_x_train = add_cluster_dims(x_train, km_train_clust[i])
                    km_x_test = add_cluster_dims(x_test, km_test_clust[i])
                    em_x_train = add_cluster_dims(x_train, em_train_clust[i])
                    em_x_test = add_cluster_dims(x_test, em_test_clust[i])
                    dr_ann(km_x_train, y_train, km_x_test, y_test,
                           [dr_step], savedir, ds, 'km')
                    dr_ann(em_x_train, y_train, em_x_test, y_test,
                           [dr_step], savedir, ds, 'em')

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
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('n_components')
    plt.ylabel('explained variance (%)')
    plt.savefig('{}/{}-pca-ev.png'.format(savedir, ds))
    plt.close('all')
    reconstruction_error = A3.recon_error(pca, X)
    logging.info('PCA reconstruction error: {}'.format(reconstruction_error))

    # second ICA
    ica = None
    best_ica_err = np.inf
    for i in range(10, 110, 10):
        ica_pipe = A3.ica(X, y, i)
        # This should be selected by kurtosis
        err = A3.recon_error(ica_pipe.named_steps['fastica'], X)
        logging.info('ICA {} components reconstruction error: {}'
                     .format(i, err))
        if err < best_ica_err:
            ica = ica_pipe.named_steps['fastica']

    # RP
    rp = None
    best_rp_err = np.inf
    for i in range(10, 160, 20):
        rp_pipe = A3.random_projection(X, y, i)
        err = A3.recon_error(rp_pipe.named_steps['gaussianrandomprojection'], X)
        logging.info('RP {} components reconstruction error: {}'
                     .format(i, err))
        if err < best_rp_err:
            rp = rp_pipe.named_steps['gaussianrandomprojection']

    logging.info('RP best n_components: {}'.format(rp.n_components_))

    # TODO: fourth dimension reduction

    return [pca, ica, rp]


def add_cluster_dims(X, clusters):
    oh = OneHotEncoder()
    c = oh.fit_transform(clusters.reshape(-1, 1)).toarray()
    return np.append(X, c, axis=1)


def dr_cluster(X, y, dr_steps, savedir, ds):
    """
    Apply dimensionality reduction and then KMeans and EM clustering
    :param X: np array, training samples
    :param y: np array, labels
    :param dr_steps: list of dimensionality reduction objects
    :param savedir: string, output directory
    :param ds: string, name of dataset
    :return: tuple, best clusters for each dr type
    """
    best_km = []
    best_em = []
    for dr_step in dr_steps:
        km, em = A3.cluster(range(2, 21), X, y,
                            savedir, ds,
                            tnse_range=None,
                            dr_step=dr_step)
        best_km.append(km[1])
        best_em.append(em[1])

    return best_km, best_em


def dr_ann(X_train, y_train, X_test, y_test, dr_steps, savedir, ds,
           cluster=None):
    if cluster:
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

    util.plotBarScores(scores, score_names, '', savedir, phaseName=ds)
    plt.close('all')


def get_args():
    parser = argparse.ArgumentParser(description='CS7641 Assignment 3')

    phases = ['cluster', 'dr', 'dr-cluster', 'dr-nn', 'dr-cluster-nn']
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
