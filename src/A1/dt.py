from sklearn.tree import DecisionTreeClassifier


def DT(pipe=False, **kwargs):
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 4, 8],
        'min_samples_split': [2],
        'max_features': [None],
        'min_impurity_decrease': [0.],
        'class_weight': [None]
    }
    prefix = 'decisiontreeclassifier' if pipe else ''
    dt = DecisionTreeClassifier(**kwargs)
    params = {prefix + k: v for k, v in params.items()}

    return dt, params
