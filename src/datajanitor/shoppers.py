from . import datajanitor
import pandas as pd


class Shoppers(datajanitor.DataJanitor):
    def __init__(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'
        super().__init__(name='shoppers', dataUrl=url, filename='shoppers.csv')

    def formatData(self, keepCorr=False, doOHE=False, **kwargs):
        df = super().formatData()

        # EDA shows high correlation between ExitRate and BounceRates
        # Drop ExitRate
        if keepCorr:
            df.drop(labels=['ExitRates'], inplace=True)

        # Encode categorical/boolean columns
        for col in ['Browser', 'Region', 'TrafficType',
                    'VisitorType', 'Month']:
            df[col] = df[col].astype('category')
        df.Revenue = df.Revenue.astype('uint8')
        df.Weekend = df.Weekend.astype('uint8')

        # save reference to numeric/categorical columns for scaling
        self.numericCols = df.select_dtypes(
            include=['int64', 'float64']).columns
        self.categoricalCols = df.select_dtypes(
            include=['category']).columns

        df_encoded = pd.get_dummies(df,
                                    columns=list(self.categoricalCols),
                                    drop_first=doOHE)
        df_encoded.drop(axis=1, columns=['Revenue', 'Weekend'], inplace=True)
        df = pd.concat([df, df_encoded], axis=1)
        df.drop(self.categoricalCols, axis=1, inplace=True)

        return df
