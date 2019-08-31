from sklearn.neural_network import MLPClassifier


def ANN(pipe=False, verbose=True, **kwargs):
    ann = MLPClassifier(verbose=verbose, **kwargs)
    prefix = 'mlpclassifier__' if pipe else ''

    params = {
        prefix + 'hidden_layer_sizes': [(50,), (100,)],
        prefix + 'activation': ['relu'],
        prefix + 'alpha': [0.0001],
        prefix + 'learning_rate_init': [0.1],
        prefix + 'solver': ['sgd']
    }
    return ann, params
