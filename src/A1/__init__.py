from .svm import SVM


def getClfParams(clfType, **kwargs):

    classifierGenerator = {
        'kernelSVM': SVM
    }

    return classifierGenerator[clfType](**kwargs)
