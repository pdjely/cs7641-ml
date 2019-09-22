from . import datajanitor
import pandas as pd


class Musk(datajanitor.DataJanitor):
    """
    Musk dataset from UCI Archive.

    Classify a molecule as being musk or not

    https://archive.ics.uci.edu/ml/datasets/musk
    https://www.openml.org/d/40666
    """
    def __init__(self):
        url = 'https://www.openml.org/data/get_csv/4965241/clean2.arff'
        super().__init__(name='musk',
                         dataUrl=url,
                         filename='clean2.csv')
        self.scoring = 'f1'

        self.gridParams = {
            'adaboost': {
                'learning_rate': [0.8, 1.0, 1.2, 1.5],
                'n_estimators': range(100, 600, 200)
            },
            'ann': {
                'hidden_layer_sizes': [(50,), (20, 20), (50, 50), (100, 100), (200, 200),
                                       (50, 50, 50), (300, 200, 100)]
            },
            'dt': {
                'max_depth': [4, 6, 12, 24]
            },
            'kernelSVM': {
                'gamma': [0.001, 0.01, 0.1]
            },
            'knn': {
                'n_neighbors': [2, 5, 10],
                'p': [1., 2., 3.]
            }
        }

    def formatData(self, keepCorr=False, doOHE=False, **kwargs):
        """
        Format musk dataset

        :param keepCorr:
        :param doOHE:
        :param kwargs:
        :return:
        """
        super().formatData()

        # musk = 1, non-musk = 0
        self.label = 'class'

        # ---- Following changes based on data analysis from pandas-profiling ----

        # drop non-descriptive columns
        self.df.drop(columns=['molecule_name', 'conformation_name'], inplace=True)
        # Categorical types already have 'object' type, so save them
        self.categoricalCols = None
        self.numericCols = self.df.columns[:-1]





