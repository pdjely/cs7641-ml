from sklearn.tree import DecisionTreeClassifier


def DT(pipe=False, **kwargs):
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 8, 16, 32, 64, 128, 256, 512],
        'max_features': [0.1, 0.25, 0.33, 0.5, 0.67, 0.8, 1.0]
    }
    prefix = 'decisiontreeclassifier' if pipe else ''
    dt = DecisionTreeClassifier(**kwargs)
    params = {prefix + k: v for k, v in params.items()}

    return dt, params
