from sklearn.svm import SVC


def SVM(pipe=False, **kwargs):
    svm = SVC(**kwargs)
    prefix = 'svc__' if pipe else ''

    gridParams = {
        prefix + 'C': [0.5, 1, 1.25],
        prefix + 'gamma': [0.0001, 0.001, 0.01, 0.1],
        prefix + 'kernel': ['rbf']
    }

    return svm, gridParams
