from .svm import SVM
from .knn import KNN


def getClfParams(clfType, **kwargs):

    classifierGenerator = {
        'kernelSVM': SVM,
        'knn': KNN
    }

    return classifierGenerator[clfType](**kwargs)
