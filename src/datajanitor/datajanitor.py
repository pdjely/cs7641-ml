import requests
import os
import tqdm
import pandas as pd

"""
Base class for retrieving, cleaning, and splitting data sets
"""
class DataJanitor:
    def __init__(self,
                 name='uninitialized',
                 dataUrl=None,
                 filename=None):
        self.name = name
        self.dataUrl = dataUrl
        self.dataLoaded = False
        self.dataDownloaded = False
        self.filestorePath = os.path.join(os.path.dirname(__file__),
                                          'datastore')
        self.filename = filename
        self.fullFilePath = os.path.join(self.filestorePath, self.filename)
        self.categoricalCols = None
        self.numericCols = None

    """"
    Retrieve dataset from local datastore or download it
    """
    def getData(self, **kwargs):
        # Now that the file is there (or should be), format it, convert to csv
        # (if necessary) and return a pandas dataframe
        self.fetchData()
        df = self.formatData(**kwargs)
        return df

    """
    Convert the raw dataset to a pandas dataframe
    """
    def formatData(self, **kwargs):
        return pd.read_csv(self.fullFilePath)

    """
    Download dataset from remote location, blasting whatever was saved locally.
    """
    def fetchData(self):
        if not (os.path.exists(self.fullFilePath)):
            print('Fetching {} dataset from {}'.format(self.name, self.dataUrl))
            # requests download adapted from stackoverflow article (1) and (2)
            # (1) https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
            # (2) https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests

            # Make request and show progressbar with download total
            r = requests.get(self.dataUrl, stream=True)
            totalSize = int(r.headers.get('content-length', 0))
            bs = 1024
            totalWritten = 0
            with open(self.fullFilePath, 'wb') as f:
                f.truncate(0)
                for chunk in tqdm.tqdm(iterable=r.iter_content(chunk_size=bs),
                                       total=totalSize // bs, unit='KB'):
                    totalWritten += len(chunk)
                    f.write(chunk)
            if totalSize and totalWritten != totalSize:
                raise(RuntimeError('Error: only downloaded {} of {}KB'.
                                   format(totalWritten, totalSize)))
            else:
                print('{} saved to local datastore'.format(self.filename))
