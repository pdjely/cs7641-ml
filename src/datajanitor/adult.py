from . import datajanitor
import pandas as pd
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.utils import compute_sample_weight


class Adult(datajanitor.DataJanitor):
    """
    Adult dataset from UCI Archive.

    Classify adults into income <=50k or >50k. Dataset is stored as csv
    without column names.

    https://archive.ics.uci.edu/ml/datasets/Adult
    """
    def __init__(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        super().__init__(name='adult',
                         dataUrl=url,
                         filename='adult.data')
        self.scoring = 'f1'

        self.gridParams = {
            'adaboost': {
                'learning_rate': [0.8, 0.9, 1.0, 1.1, 1.2],
                'n_estimators': range(50, 300, 50)
            },
            'ann': {
                'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
                'hidden_layer_sizes': [(20, 20), (50, 50), (100, 100), (200, 200),
                                       (20, 20, 20), (50, 50, 50), (300, 200, 100)]
            },
            'dt': {
                'max_depth': range(3, 8),
                'max_leaf_nodes': range(5, 8)
            },
            'kernelSVM': {
                'C': [0.25, 0.50, 0.75],
                'gamma': [0.001, 0.01, 0.1]
            },
            'knn': {
                'n_neighbors': [2, 10, 20, 50]
            }
        }

    def formatData(self, keepCorr=False, doOHE=False, **kwargs):
        """
        Format adult dataset

        :param keepCorr:
        :param doOHE:
        :param kwargs:
        :return:
        """
        super().formatData()

        # rename columns
        self.df.columns = [
            'age',
            'workclass',
            'finalWeight',
            'education',
            'yearsEducation',
            'maritalStatus',
            'occupation',
            'familyRole',
            'race',
            'sex',
            'capitalGain',
            'capitalLoss',
            'hoursPerWeek',
            'nativeCountry',
            'income50k'
        ]

        # label is in income50k and is a text value. convert to 1/0
        self.df.income50k.replace((' <=50K', ' >50K'), (0, 1), inplace=True)
        self.label = 'income50k'

        # ---- Following changes based on data analysis from pandas-profiling ----

        # Drop irrelevant columns
        self.df.drop(labels=['finalWeight', 'education'], axis=1, inplace=True)

        # change label to uint8
        self.df.income50k = self.df.income50k.astype('uint8')

        # Categorical types already have 'object' type, so save them
        self.categoricalCols = self.df.select_dtypes(
            include=['category', 'object']).columns
        for col in self.categoricalCols:
            self.df[col] = self.df[col].astype('category')
        self.numericCols = self.df.select_dtypes(
            include=['int64']).columns

        df_encoded = pd.get_dummies(self.df,
                                    columns=list(self.categoricalCols),
                                    drop_first=doOHE)
        self.df = df_encoded
