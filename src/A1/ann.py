from sklearn.neural_network import MLPClassifier


def ANN(pipe=False, verbose=False, **kwargs):
    ann = MLPClassifier(activation='relu',
                        solver='adam',
                        verbose=verbose,
                        early_stopping=True,
                        shuffle=True,
                        random_state=10,
                        max_iter=5000,
                        learning_rate='adaptive',
                        **kwargs)
    # prefix = 'mlpclassifier__' if pipe else ''
    prefix = 'classifier__' if pipe else ''

    # Used for fine-tuning model based on validation curves
    params = {
        prefix + 'hidden_layer_sizes': [(10,), (20,), (50,), (200,),
                                        (10, 10), (20, 20), (50, 50),
                                        (10, 10, 10), (20, 20, 20), (50, 50, 50),
                                        (200, 200, 200), (200, 100),
                                        (200, 100, 50)],
        prefix + 'alpha': [0.0001, 0.001]
    }
    return ann, params
