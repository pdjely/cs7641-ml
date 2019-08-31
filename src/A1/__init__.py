from .svm import SVM
from .knn import KNN
from .ann import ANN


def getClfParams(clfType, **kwargs):

    classifierGenerator = {
        'kernelSVM': SVM,
        'knn': KNN,
        'ann': ANN
    }

    return classifierGenerator[clfType](**kwargs)
