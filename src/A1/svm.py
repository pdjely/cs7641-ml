from sklearn.svm import SVC


def SVM(pipe=False, **kwargs):
    svm = SVC(**kwargs)
    prefix = 'svc__' if pipe else ''

    gridParams = {
        prefix + 'C': [0.95, 1.0],
        prefix + 'gamma': [0.001, 0.01],
        prefix + 'kernel': ['rbf', 'linear']
    }

    return svm, gridParams
