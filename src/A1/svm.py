from sklearn.svm import SVC


def SVM(**kwargs):
    svm = SVC(**kwargs)

    gridParams = {
        'C': [0.5, 1, 1.25],
        'gamma': [0.0001, 0.001, 0.01, 0.1],
        'kernel': ['rbf']
    }

    return svm, gridParams
