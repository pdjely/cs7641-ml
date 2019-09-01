from sklearn.neural_network import MLPClassifier


def ANN(pipe=False, verbose=False, **kwargs):
    ann = MLPClassifier(verbose=verbose, **kwargs)
    prefix = 'mlpclassifier__' if pipe else ''

    params = {
        prefix + 'hidden_layer_sizes': [(100,)],
        prefix + 'activation': ['relu', 'tanh'],
        prefix + 'alpha': [0.0001, 0.001],
        prefix + 'learning_rate_init': [0.1, 0.01, 0.001],
        prefix + 'solver': ['sgd'],
        prefix + 'beta_1': [0.8, 0.9],
        prefix + 'beta_2': [0.9, 0.99],
        prefix + 'epsilon': [1e-8]
    }
    return ann, params
