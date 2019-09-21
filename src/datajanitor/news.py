from . import datajanitor
import pandas as pd
import os
import zipfile


class News(datajanitor.DataJanitor):
    def __init__(self):
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'
        super().__init__(name='news',
                         dataUrl=url,
                         filename='OnlineNewsPopularity.zip')

        # Standard file fetch gets the zip file. we have to unzip it after fetching
        self.csvfile = os.path.join(self.filestorePath,
                                    'OnlineNewsPopularity',
                                    'OnlineNewsPopularity.csv')
        self.scoring = 'f1_weighted'

        self.gridParams = {
            'adaboost': {
                'learning_rate': [0.6, 1.0, 1.2, 1.5, 2.0]
            },
            'ann': {
                'alpha': [1e-4, 1e-2, 1e-1, 1.0, 1.2],
                'hidden_layer_sizes': [(50, 50), (200, 200), (50, 50, 50),
                                       (300, 200, 100)]
            },
            'dt': {
                'max_depth': range(4, 8),
                'max_leaf_nodes': range(5, 10)
            },
            'kernelSVM': {
                'C': [1.0, 1.2, 1.5]
            },
            'knn': {
                'n_neighbors': [70, 90, 120, 150],
                'p': [0.01, 0.1, 1.0]
            }
        }

    def fetchData(self):
        super().fetchData()

        # Unzip stolen from Python stdlib example
        # https://docs.python.org/3/library/zipfile.html
        with zipfile.ZipFile(self.fullFilePath, mode='r') as myzip:
            myzip.extract('OnlineNewsPopularity/OnlineNewsPopularity.csv',
                          path=self.filestorePath)

    def formatData(self, keepCorr=False, doOHE=False, **kwargs):
        # Don't call super. Instead open saved csv file which is extracted
        # after fetch
        self.df = pd.read_csv(self.csvfile)

        # Generate target label, fix colnames, and drop unneeded columns
        self.df.columns = [x.strip() for x in self.df.columns]
        self.df['label'] = (self.df['shares'] >= 1400).astype('uint8')
        self.df.drop(columns=['url', 'timedelta', 'shares'], inplace=True)

        if not keepCorr:
            self.df.drop(columns=['n_non_stop_words', 'n_non_stop_unique_tokens'],
                         inplace=True)

        # Store label column
        self.label = 'label'

        # save reference to numeric/categorical columns for scaling
        self.numericCols = self.df.select_dtypes(
            include=['int64', 'float64']).columns
        self.categoricalCols = self.df.select_dtypes(
            include=['category']).columns
