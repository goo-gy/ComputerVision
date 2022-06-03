import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn import inspection

l_pRandomState = [20, 30, 40]
l_pC = [10, 1, 0.1]


def prob1():
    for row, pRandomState in enumerate(l_pRandomState):
        coords, labels = datasets.make_blobs(
            n_samples=100, cluster_std=1.2, random_state=pRandomState, centers=2)

        plt.figure(figsize=(15, 5))
        plt.suptitle('random_state=%d' % pRandomState)
        for col, pC in enumerate(l_pC):
            plt.subplot(1, 3, col + 1)
            clf = SVC(kernel="linear", C=pC)
            result = clf.fit(coords, labels)

            ax = plt.gca()
            ax.set_title('C=%.1f' % pC)
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


prob1()
