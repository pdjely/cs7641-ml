from sklearn.neighbors import KNeighborsClassifier


def KNN(dsname, pipe=False, **kwargs):
    knn = KNeighborsClassifier(**kwargs)
    # prefix = 'knn__' if pipe else ''
    prefix = 'classifier__' if pipe else ''

    params = {
        prefix + 'n_neighbors': [2, 5, 10, 20, 50, 100],
        prefix + 'weights': ['distance', 'uniform'],
    }

    return knn, params
