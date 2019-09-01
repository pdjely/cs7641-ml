from sklearn.neighbors import KNeighborsClassifier


def KNN(pipe=False, **kwargs):
    knn = KNeighborsClassifier(**kwargs)
    prefix = 'knn__' if pipe else ''

    params = {
        prefix + 'n_neighbors': [5, 10],
        prefix + 'weights': ['distance', 'uniform'],
        prefix + 'algorithm': ['ball_tree', 'kd_tree'],
        prefix + 'leaf_size': [30]
    }

    return knn, params
