import requests
import os
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


class DataJanitor:
    """
    Base class for retrieving, cleaning, and splitting data sets

    The class will automatically handle splitting the dataset into train,
    validation, and test sets, along with fitting a scaling classifier
    to the train set.
    """
    def __init__(self,
                 name='uninitialized',
                 dataUrl=None,
                 filename=None,
                 **kwargs):
        self.name = name
        self.dataUrl = dataUrl
        self.filestorePath = os.path.join(os.path.dirname(__file__),
                                          'datastore')
        self.filename = filename
        self.fullFilePath = os.path.join(self.filestorePath, self.filename)
        self.X = None
        self.y = None

        # These need to be set in subclasses
        self.categoricalCols = None
        self.numericCols = None
        self.label = None
        self.df = None  # raw pandas dataframe
        self.scoring = 'accuracy'
        self.gridParams = None

    def getData(self, **kwargs):
        # Download file (if necessary), format it, convert to csv
        if self.df is None:
            self.fetchData()
        self.formatData(**kwargs)

        # dataframe has been loaded and formatted, now save just the
        # numpy arrays
        self.X = self.df.drop(self.label, axis=1).values
        self.y = self.df[self.label].values

    def getDataFrame(self, **kwargs):
        """"
        Retrieve underlying dataset as a pandas dataframe

        Fetch the dataset and format it if necessary.
        """
        self.getData(**kwargs)

        return self.df

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
                             stratify=self.df[self.label])

        return trainx, testx, trainy, testy

    def formatData(self, **kwargs):
        """
        Convert the raw dataset to a pandas dataframe

        Subclasses are expected to call this method at the start of the
        overriding method to convert the downloaded file to a pandas
        data frame
        """
        self.df = pd.read_csv(self.fullFilePath)

    def fetchData(self):
        """
        Download dataset from remote location, blasting whatever was saved locally.
        """
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

    def getScorer(self):
        return self.scoring

    def getGridParams(self, clfType, pipe=True):
        """
        Grid params are defined in child class constructors
        """
        if pipe:
            params = {'classifier__' + k: v for k, v
                      in self.gridParams[clfType].items()}
            return params
        else:
            return self.gridParams[clfType]
