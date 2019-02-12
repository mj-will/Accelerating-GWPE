
import numpy as np
from gwfa.utils.general import fuzz

def check_normalise(x):
    if np.isclose(np.sum(x), 1.):
        return x
    else:
        return x / np.sum(x)

def kullback_leibler_divergence(P, Q):
    """
    Compute the Kullback-Leibler divergence for two distributions.
    """
    P = check_normalise(P) + fuzz()
    Q = check_normalise(Q) + fuzz()
    return np.sum(P * np.log(P / Q))

def jenson_shannon_divergence(P, Q):
    """Compute the Jenson-Shannon divergence"""
    M = 0.5 * (P + Q)
    return 0.5 * (kullback_leibler_divergence(P, M)) + 0.5 * (kullback_leibler_divergence(Q, M))

def mean_squared_error(y_true, y_pred):
    """Compute the mean squared error"""
    return np.mean((y_true - y_pred) ** 2.)

def max_squared_error(y_true, y_pred):
    """Compute the maximum squared error"""
    return np.max((y_true - y_pred) ** 2.)
