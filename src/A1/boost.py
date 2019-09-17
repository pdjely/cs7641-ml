from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def AdaBoost(dsname, pipe=False, **kwargs):
    boost = AdaBoostClassifier(**kwargs)
    # prefix = 'adaboostclassifier__' if pipe else ''
    prefix = 'classifier__' if pipe else ''

    params = {
        prefix + 'n_estimators': [10, 30, 50, 100, 200, 500],
        prefix + 'learning_rate': [1.2, 1., 0.8]
    }

    return boost, params
