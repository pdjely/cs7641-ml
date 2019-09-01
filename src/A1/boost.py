from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def AdaBoost(pipe=False, **kwargs):
    boost = AdaBoostClassifier(**kwargs)
    prefix = 'adaboostclassifier__' if pipe else ''

    params = {
        prefix + 'base_estimator': [DecisionTreeClassifier(max_depth=1),
                                    DecisionTreeClassifier(max_depth=2)],
        prefix + 'n_estimators': [50],
        prefix + 'learning_rate': [1.2, 1., 0.8]
    }

    return boost, params
