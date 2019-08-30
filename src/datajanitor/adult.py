from . import datajanitor

class Adult(datajanitor.DataJanitor):
    """
    Adult dataset from UCI Archive.

    Classify adults into income <50k or >=50k

    https://archive.ics.uci.edu/ml/datasets/Adult
    """
    def __init__(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        super().__init__(name='adult', dataUrl=url)

    def formatData(self, keepCorr=False, doOHE=False, **kwargs):
        """
        Format adult dataset

        :param keepCorr:
        :param doOHE:
        :param kwargs:
        :return:
        """
        super().formatData()


