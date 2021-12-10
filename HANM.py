#! -*- coding: utf-8 -*-
import math
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from Synthetic_data import create_data

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


def HANM_V(data):
    # load data
    cidata = data
    N = cidata.shape[0]
    D = cidata.shape[1]
    x_train = cidata[:, :(D - 1)]
    y_train = cidata[:, D - 1]
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    batch_size = N
    original_dim = D - 1
    latent_dim = 1
    h1 = 15
    h2 = 7
    h3 = 15
    epochs = 50

    x = Input(shape=(original_dim,))  # N-15-7-15-1
    y = Input(shape=(1,))
    m1 = Dense(h1, activation='relu',
               kernel_initializer='random_uniform',
               bias_initializer='zeros')(x)
    m2 = Dense(h2, activation='relu')(m1)
    m3 = Dense(h3, activation='relu')(m2)

    z_mean = Dense(latent_dim)(m3)
    z_log_var = Dense(latent_dim)(m3)

    # reparameterization 
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    d1 = h3
    d2 = h2
    d3 = h1

    t1 = Dense(d1, activation='relu')(z)  # 1-15-7-15-N
    t2 = Dense(d2, activation='relu')(t1)
    t3 = Dense(d3, activation='relu')(t2)
    x_hat = Dense(original_dim, activation='sigmoid')(t3)

    model = Model([x, y], x_hat)

    rec_loss = K.mean(K.square(x - x_hat), axis=-1)
    func_loss = K.mean(K.square(y - z), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    model_loss = K.mean(rec_loss + func_loss + kl_loss)

    model.add_loss(model_loss)
    model.compile(optimizer='rmsprop')

    model.fit([x_train, y_train],
            shuffle=True,
            epochs=epochs,
            verbose=0,
            batch_size=batch_size
            )

    e1 = Model(x, z_mean)
    e2 = Model(x, z_log_var)
    x_test_e1 = e1.predict(x_train, batch_size=batch_size)
    x_test_e2 = e2.predict(x_train, batch_size=batch_size)
    z = Lambda(sampling, output_shape=(latent_dim,))([x_test_e1, x_test_e2])
    z = K.eval(z)
    return z


def HANM(data, seed=0):
    """
    :param data: Data of many-to-one causality {Xi,...,Xn,Y}.
    :param seed: The random seed.
    :return: The causality of Xi and Y. 0: Y->Xi; 1: Xi->Y; -1: Non-identifiable.

    HANM algorithm.

    **Description**: Hierarchical Additive Noise Model (HANM) generalizes 
    many-to-one causality into an approximate pair relationship to identify 
    the causal relationship.

    **Data Type**: Continuous

    Example:

        >>> data = create_data(n_causes=8, n_samples=500, sigma=0.0025, func='exp', seed=0)
        >>> result = HANM(data)
        
    """

    cidata = data
    D = cidata.shape[1]
    x = cidata[:, :(D - 1)]
    y = cidata[:, D - 1]
    result = np.zeros(D - 1)

    tf.random.set_seed(seed)
    # forward
    zy = HANM_V(data)
    y = y.reshape(-1, )
    zy = zy.reshape(-1, )
    f1 = np.polyfit(zy, y, 3)
    p1 = np.poly1d(f1)
    nx = p1(zy) - y
    zy = zy.reshape(-1, 1)
    nx = nx.reshape(-1, 1)
    x_to_y = HSIC(zy, nx)
    # backward
    for i in range(0, D - 1):
        y = y.reshape(-1, )
        x_t = x[:, i].reshape(-1, )
        f2 = np.polyfit(y, x_t, 3)
        p2 = np.poly1d(f2)
        ny = p2(y) - x_t
        y = y.reshape(-1, 1)
        ny = ny.reshape(-1, 1)
        y_to_x = HSIC(y, ny)
        delta = 0.05 * min(x_to_y, y_to_x)
        if x_to_y + delta < y_to_x:
            result[i] = 1
        elif y_to_x + delta < x_to_y:
            result[i] = 0
        else:
            result[i] = -1
    return result


if __name__ == "__main__":
    data = create_data(n_causes=8, n_samples=500, sigma=0.0025, func='exp', seed=0)
    result = HANM(data)
    print(result)
