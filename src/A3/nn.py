from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging
import timeit


def ann_factory(dsname):
    hidden_layers = {
        'musk': (200, 200),
        'shoppers': (20, 20, 20)
    }

    ann = MLPClassifier(solver='adam',
                        early_stopping=True,
                        shuffle=True,
                        random_state=10,
                        learning_rate='adaptive',
                        hidden_layer_sizes=hidden_layers[dsname])
    logging.info('Created MLPClassifier with parameters: {}'
                 .format(ann.get_params()))
    return ann


def dr_ann(X, y, dr_step, ds):
    ann = ann_factory(ds)
    pipe = make_pipeline(StandardScaler(),
                         dr_step,
                         ann)
    start_time = timeit.default_timer()
    pipe.fit(X, y)
    train_time = timeit.default_timer() - start_time
    logging.info('Training took {:.4} s'.format(train_time))

    return pipe


def baseline_ann(X, y, ds):
    ann = ann_factory(ds)
    print(y.shape)
    pipe = make_pipeline(StandardScaler(),
                         ann)
    start_time = timeit.default_timer()
    pipe.fit(X, y)
    train_time = timeit.default_timer() - start_time
    logging.info('Training took {:.4} s'.format(train_time))

    return pipe
