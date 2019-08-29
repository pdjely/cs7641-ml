from . import datajanitor
import pandas as pd


class Shoppers(datajanitor.DataJanitor):
    def __init__(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'
        super().__init__(name='shoppers', dataUrl=url, filename='shoppers.csv')

    def formatData(self, keepCorr=False, doOHE=False, **kwargs):
        super().formatData()

        # EDA shows high correlation between ExitRate and BounceRates
        # Drop ExitRate
        if keepCorr:
            self.df.drop(labels=['ExitRates'], inplace=True)

        # Encode categorical/boolean columns
        for col in ['Browser', 'Region', 'TrafficType', 'OperatingSystems',
                    'VisitorType', 'Month']:
            self.df[col] = self.df[col].astype('category')
        self.df.Revenue = self.df.Revenue.astype('uint8')
        self.df.Weekend = self.df.Weekend.astype('uint8')

        # Store label column
        self.label = 'Revenue'

        # save reference to numeric/categorical columns for scaling
        self.numericCols = self.df.select_dtypes(
            include=['int64', 'float64']).columns
        self.categoricalCols = self.df.select_dtypes(
            include=['category']).columns

        df_encoded = pd.get_dummies(self.df,
                                    columns=list(self.categoricalCols),
                                    drop_first=doOHE)
        # df_encoded.drop(axis=1, columns=['Revenue', 'Weekend'], inplace=True)
        # self.df = pd.concat([self.df, df_encoded], axis=1)
        # self.df.drop(self.categoricalCols, axis=1, inplace=True)
        self.df = df_encoded
