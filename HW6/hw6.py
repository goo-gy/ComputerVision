import numpy as np
import matplotlib.pyplot as plt

img1 = plt.imread('./data/warrior_a.jpg')
img2 = plt.imread('./data/warrior_b.jpg')

cor1 = np.load("./data/warrior_a.npy")
cor2 = np.load("./data/warrior_b.npy")

colors = ['tab:blue',
          'tab:orange',
          'tab:green',
          'tab:red',
          'tab:purple',
          'tab:brown',
          'tab:pink',
          'tab:gray',
          'tab:olive',
          'tab:cyan',
          'lime']


def compute_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    F = None
    # YOUR CODE BEGINS HERE

    # build matrix for equations in Page 51
    A = np.empty((0, 9))
    for k in range(n):
        row = np.outer(x1.T[k], x2.T[k]).flatten()
        A = np.append(A, [row], axis=0)
    # compute the solution in Page 51
    # SVD를 사용하여 최소화 eigenvector를 구함
    F = np.linalg.svd(A)[2][-1]
    F = F.reshape((3, 3))

    # constrain F: make rank 2 by zeroing out last singular value (Page 52)
    # SVD를 사용하여 분리
    U, sigma, V_t = np.linalg.svd(F)
    sigma = np.diag(sigma)
    sigma[2][2] = 0
    F = U @ sigma @ V_t  # F = U.dot(sigma).dot(V_t)

    # YOUR CODE ENDS HERE

    return F


def compute_norm_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1*mean_1[0]], [0, S1, -S1*mean_1[1]], [0, 0, 1]])
    x1 = T1 @ x1

    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2*mean_2[0]], [0, S2, -S2*mean_2[1]], [0, 0, 1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)

    # reverse normalization
    F = T2.T @ F @ T1

    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    # YOUR CODE BEGINS HERE
    # SVD를 사용하여 최소화 eigenvector를 구함
    e1 = np.linalg.svd(F)[2][-1]
    e2 = np.linalg.svd(F.T)[2][-1]
    # e1 = np.linalg.eig(F)[0]
    # e2 = np.linalg.eig(F.T)[0]
    e1 = e1 / e1[-1]
    e2 = e2 / e2[-1]
    # YOUR CODE ENDS HERE

    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    # YOUR CODE BEGINS HERE
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    X = np.linspace(0, img1.shape[0], img1.shape[0] + 1)
    for i, p1 in enumerate(cor1.T):
        color = colors[i % len(colors)]
        gradient = (e1[1] - p1[1]) / (e1[0] - p1[0])
        intercept = e1[1] - gradient * e1[0]
        Y = gradient * X + intercept
        plt.scatter(p1[0], p1[1], color=color)
        plt.plot(X, Y, color=color)
    # ------------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    X = np.linspace(0, img2.shape[0], img2.shape[0] + 1)
    for i, p2 in enumerate(cor2.T):
        color = colors[i % len(colors)]
        gradient = (e2[1] - p2[1]) / (e2[0] - p2[0])
        intercept = e2[1] - gradient * e2[0]
        Y = gradient * X + intercept
        plt.scatter(p2[0], p2[1], color=color)
        plt.plot(X, Y, color=color)
    plt.show()
    # YOUR CODE ENDS HERE

    return


draw_epipolar_lines(img1, img2, cor1, cor2)
