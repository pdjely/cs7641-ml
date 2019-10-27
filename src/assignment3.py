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
import A1


def main():
    args = get_args()
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
        pca = A3.pca(x_train, y_train)
        plt.savefig('{}/pca.png'.format(savedir))
        np.savetxt('{}/{}-pca-ev.csv'.format(savedir, ds),
                   pca.named_steps['pca'].explained_variance_)

        # *********************** #
        # **** DR + Cluster  **** #
        # *********************** #

        # make pipeline with best pca and best cluster number
        ann, _ = A1.getClfParams('ann')
        ann.set_params(hidden_layer_sizes=(200, 200))
        pipe = make_pipeline(StandardScaler(), PCA(0.95), ann)
        pipe.fit(x_train, y_train)





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
