#! -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn.decomposition import PCA


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    Q = I - unit / n
    return np.dot(np.dot(Q, K), Q)


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / sigma / sigma
    np.exp(KX, KX)
    return KX


def HSIC(X, Y):
    return np.sum(centering(rbf(X)) * centering(rbf(Y)))


path = './Multi-reason-nsin/8/500/'
file = os.listdir(path)
anm_c = 0
anm_w = 0

for f in file:
    data = path + f

    cidata = np.loadtxt(path + f)
    N = cidata.shape[0]
    D = cidata.shape[1]

    x = cidata[:, :(D - 1)]
    y = cidata[:, D - 1]

    pca = PCA(n_components=1)
    z = pca.fit_transform(x)

    z = np.array(z)

    z = z.reshape(-1, )
    y = y.reshape(-1, )

    f1 = np.polyfit(z, y, 3)
    p1 = np.poly1d(f1)
    nx = p1(z) - y
    z = z.reshape(-1, 1)
    nx = nx.reshape(-1, 1)
    x_to_y = HSIC(z, nx)

    for i in range(0, D - 1):
        y = y.reshape(-1, )
        x_t = x[:, i].reshape(-1, )
        f2 = np.polyfit(y, x_t, 3)
        p2 = np.poly1d(f2)
        ny = p2(y) - x_t
        y = y.reshape(-1, 1)
        ny = ny.reshape(-1, 1)
        y_to_x = HSIC(y, ny)

        if x_to_y < y_to_x:
            print('result:x->y', 'x->y:', x_to_y, ' y->x:', y_to_x)
            anm_c += 1
        else:
            print('result:y->x', 'x->y:', x_to_y, ' y->x:', y_to_x)
            anm_w += 1
    print(anm_c + anm_w, anm_c / (anm_c + anm_w))
