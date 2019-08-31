from .svm import SVM
from .knn import KNN
from .ann import ANN
from .boost import AdaBoost
from .dt import DT


def getClfParams(clfType, **kwargs):

    classifierGenerator = {
        'kernelSVM': SVM,
        'knn': KNN,
        'ann': ANN,
        'adaboost': AdaBoost,
        'dt': DT
    }

    return classifierGenerator[clfType](**kwargs)
