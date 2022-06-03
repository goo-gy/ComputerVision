import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn import inspection

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

l_pRandomState = [20, 30, 40]
l_pC1 = [10, 1, 0.1]
l_pC2 = [10, 0.1]


def prob1():
    for row, pRandomState in enumerate(l_pRandomState):
        coords, labels = datasets.make_blobs(
            n_samples=100, cluster_std=1.2, random_state=pRandomState, centers=2)

        plt.figure(figsize=(15, 5))
        for col, pC in enumerate(l_pC1):
            plt.subplot(1, 3, col + 1)
            ax = plt.gca()
            ax.set_title('C=a%.1f' % pC)

            clf = SVC(kernel="linear", C=pC)
            clf.fit(coords, labels)
            inspection.DecisionBoundaryDisplay.from_estimator(
                clf,
                coords,
                plot_method="contour",
                levels=[-1, 0, 1],
                linestyles=["--", "-", "--"],
                ax=ax,
            )

            plt.scatter(coords[labels == 0, 0], coords[labels == 0, 1])
            plt.scatter(coords[labels == 1, 0], coords[labels == 1, 1])
            plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[
                :, 1], s=250, alpha=0.3)
        plt.show()


def createDatasets():
    coords, labels = datasets.make_circles(
        n_samples=100, factor=0.1, noise=0.1)

    plt.scatter(coords[labels == 0, 0], coords[labels == 0, 1])
    plt.scatter(coords[labels == 1, 0], coords[labels == 1, 1])
    plt.show()
    return [coords, labels]


def gauss_rbf(coords):
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = np.exp(-(X**2 + Y**2))
    return X, Y, Z


def kernelFunction(coords, labels):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = gauss_rbf(coords)

    ax.scatter(X[labels == 0], Y[labels == 0], Z[labels == 0])
    ax.scatter(X[labels == 1], Y[labels == 1], Z[labels == 1])
    plt.show()


def trainSVM(coords, labels):
    plt.figure(figsize=(10, 5))
    for col, pC in enumerate(l_pC2):
        plt.subplot(1, 2, col + 1)
        ax = plt.gca()
        ax.set_title('C=%.1f' % pC)

        clf = SVC(kernel="rbf", C=10)
        clf.fit(coords, labels)
        inspection.DecisionBoundaryDisplay.from_estimator(
            clf,
            coords,
            plot_method="contour",
            levels=[-1, 0, 1],
            linestyles=["--", "-", "--"],
            ax=ax,
        )

        plt.scatter(coords[labels == 0, 0], coords[labels == 0, 1])
        plt.scatter(coords[labels == 1, 0], coords[labels == 1, 1])
    plt.show()


def prob2():
    coords, labels = createDatasets()
    kernelFunction(coords, labels)
    trainSVM(coords, labels)


if(__name__ == "__main__"):
    prob1()
    prob2()
