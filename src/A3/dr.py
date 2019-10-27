from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA


def pca(X, y, n_components=0.95):
    pipe = make_pipeline(StandardScaler(), PCA(n_components=n_components))
    reduced = pipe.fit_transform(X)
    print('pca found', pipe.named_steps['pca'].n_components_, ' components')
    plt.scatter(reduced[y == 0, 0], reduced[y == 0, 1])
    plt.scatter(reduced[y == 1, 0], reduced[y == 1, 1])

    return pipe


def ica(X, y):
    pipe = make_pipeline(StandardScaler(), FastICA(max_iter=500))
    reduced = pipe.fit_transform(X)
    print('Component shape:', pipe.named_steps['fastica'].components_.shape)
    print(pipe.named_steps['fastica'].components_)

    return pipe
