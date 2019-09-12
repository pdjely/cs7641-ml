from sklearn.svm import SVC


def SVM(pipe=False, **kwargs):
    svm = SVC(**kwargs)
    prefix = 'svc__' if pipe else ''

    gridParams = {
        prefix + 'C': [0.001, 0.01, 0.1, 1.0, 1.2, 1.5],
        prefix + 'gamma': [0.001, 0.01, 0.1],
        prefix + 'kernel': ['rbf', 'linear']
    }

    return svm, gridParams
