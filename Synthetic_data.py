#! -*- coding: utf-8 -*-
import numpy as np


def create_data(n_causes=8, n_samples=500, sigma=0.0025, func='exp', seed=0):
    """
    :param n_causes: Number of causes for many-to-one causality.
    :param n_samples: Sample size for many-to-one causality.
    :param sigma: Variance for additive noise.
    :param func: Generation function for synthetic data including linear, sin and exp.
    :param seed: The random seed.
    :return: Data of many-to-one causality {Xi,...,Xn,Y}

    Generation of synthetic data in Hierarchical Additive Noise Model (HANM).

    **Description**: The data is generated into two categories with a total of three groups.
    The first category is a linear causal relationship, and the second is a nonlinear causal
    relationship consisting of trigonometric functions and exponential functions.

    **Data Type**: Continuous

    Example:

        >>> data = create_data(n_causes=8, n_samples=500, sigma=0.0025, func='exp', seed=0)

    """
    np.random.seed(seed)

    W = np.random.uniform(-1, 1, n_samples)
    X = np.zeros([n_samples, n_causes])
    Y = np.zeros(n_samples)

    if func == 'linear':
        for i in range(0, n_causes):
            X[:, i] = np.random.normal(0, 1) * np.power(W, 3) + np.random.normal(0, 1) * W + np.random.normal(0, 1) + np.random.normal(0, sigma, n_samples)
            Y += np.random.uniform(0.1, 1) * X[:, i]
    elif func == 'sin':
        for i in range(0, n_causes):
            X[:, i] = np.random.normal(0, 1) * np.power(W, 3) + np.random.normal(0, 1) * W + np.random.normal(0, 1) + np.random.normal(0, sigma, n_samples)
            Y += np.random.uniform(0.1, 1) * np.sin(X[:, i])
    elif func == 'exp':
        for i in range(0, n_causes):
            X[:, i] = np.random.normal(0, 1) * np.power(W, 3) + np.random.normal(0, 1) * W + np.random.normal(0, 1) + np.random.normal(0, sigma, n_samples)
            Y += np.random.uniform(0.1, 1) * np.exp(-X[:, i])
    Y += np.random.normal(0, sigma, n_samples)

    data = np.c_[X, Y]

    return data


if __name__ == "__main__":
    data = create_data(n_causes=8, n_samples=500, sigma=0.0025, func='exp', seed=1)
