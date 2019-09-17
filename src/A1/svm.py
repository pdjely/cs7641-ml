from sklearn.svm import SVC


def SVM(dsname, pipe=False, **kwargs):
    kwargs['max_iter'] = kwargs.get('max_iter', 5000)
    svm = SVC(**kwargs)
    # prefix = 'svc__' if pipe else ''
    prefix = 'classifier__' if pipe else ''

    gridParams = {
        prefix + 'C': [0.95, 1.0],
        prefix + 'gamma': [0.001, 0.01],
        prefix + 'kernel': ['linear', 'rbf']
    }

    return svm, gridParams
