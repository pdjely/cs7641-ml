import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
import datajanitor
import util
import A3

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline


def main():
    args = getArgs()
    savedir = util.mktmpdir(args.outdir)

    for ds in ['musk']:
        dataset = datajanitor.getDataset(ds)
        dataset.getData()
        x_train, x_test, y_train, y_test = \
            dataset.partitionData(percent=0.3, randomState=10)

        # ********************* #
        # **** Clustering  **** #
        # ********************* #
        if 'cluster' in args.phase:
            km, em = A3.cluster(range(2, 21), x_train, y_train, savedir, ds)

        # ************************ #
        # **** Dim Reduction  **** #
        # ************************ #

        # PCA
        pca = make_pipeline(StandardScaler(), PCA(0.95))
        reduced = pca.fit_transform(x_train)
        print('pca found', pca.named_steps['pca'].n_components_, ' components')
        plt.scatter(reduced[y_train == 0, 0], reduced[y_train == 0, 1])
        plt.scatter(reduced[y_train == 1, 0], reduced[y_train == 1, 1])
        plt.savefig('{}/pca.png'.format(savedir))
        np.savetxt('{}/{}-pca-ev.csv'.format(savedir, ds),
                   pca.named_steps['pca'].explained_variance_)

        # ICA
        # ica = make_pipeline(StandardScaler(), FastICA(max_iter=500))
        # reduced = ica.fit_transform(x_train)
        # print('Component shape:', ica.named_steps['fastica'].components_.shape)
        # print(ica.named_steps['fastica'].components_)


def getArgs():
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
