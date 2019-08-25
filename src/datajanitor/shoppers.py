from . import datajanitor

class Shoppers(datajanitor.DataJanitor):
    def __init__(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'
        super().__init__(name='shoppers', dataUrl=url, filename='shoppers.csv')
