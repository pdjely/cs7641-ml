from sklearn.tree import DecisionTreeClassifier


def DT(pipe=False, **kwargs):
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None],
        'min_samples_split': [2],
        'max_features': [None]
    }
    prefix = 'decisiontreeclassifier' if pipe else ''
    dt = DecisionTreeClassifier(**kwargs)
    params = {prefix + k: v for k, v in params.items()}

    return dt, params
