from .svm import SVM
from .knn import KNN
from .ann import ANN
from .boost import AdaBoost
from .dt import DT


def getClfParams(clfType, dsname=None, **kwargs):

    classifierGenerator = {
        'kernelSVM': SVM,
        'knn': KNN,
        'ann': ANN,
        'adaboost': AdaBoost,
        'dt': DT
    }

    return classifierGenerator[clfType](dsname=dsname, **kwargs)
