from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from scipy.special import comb
import matplotlib.pyplot as plt
import numpy as np
import math

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_examples, n_features]

    gamma: float
        Tunning parameter of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    ---------
    alphas: {NumPy ndarray}, shape = [n_examples, k_features]
        Project dataset

    lambdas: list
    """
    # calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N))
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected examples)
    alphas = np.column_stack([eigvecs[: ,i] for i in range(n_components)])

    # Collect the corresponding eigenvalues
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas, lambdas

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


def emsemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k) for k in range(k_start, n_classifier + 1)]
    return sum(probs)
