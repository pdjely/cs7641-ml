from . import datajanitor
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class Cancer(datajanitor.DataJanitor):
    def __init__(self):
        super().__init__(name='cancer',
                         dataUrl='',
                         filename='')
        self.scoring = 'f1_weighted'

        # very simple dataset so grid params not very thorough
        self.gridParams = {
            'adaboost': {
                'learning_rate': [0.8, 1.2],
                'n_estimators': range(50, 150, 50)
            },
            'ann': {
                'hidden_units': [(20, 20), (50, 50),
                                 (20, 20, 20), (50, 50, 50)]
            },
            'dt': {
                'max_depth': range(3, 20, 5)
            },
            'kernelSVM': {
                'C': [0.25, 0.50, 0.75],
                'gamma': [0.001, 0.01, 0.1]
            },
            'knn': {
                'n_neighbors': [2, 10, 20, 50]
            }
        }

    def getData(self, **kwargs):
        # Download file (if necessary), format it, convert to csv
        if self.df is None:
            self.X, self.y = load_breast_cancer(return_X_y=True)
            self.df = pd.DataFrame(self.X)

        self.formatData(**kwargs)

    def formatData(self, keepCorr=False, doOHE=False, **kwargs):
        pass

    def partitionData(self, percent=0.3, randomState=1):
        """
        Split dataset into train and validation, and test sets
        :param randomState:
        :param percent: percent of examples for validation set
        :return: train_x, test_x, train_y, test_y
        """
        trainx, testx, trainy, testy = \
            train_test_split(self.X,
                             self.y,
                             test_size=percent,
                             shuffle=True,
                             random_state=randomState,
                             stratify=self.y)

        return trainx, testx, trainy, testy
