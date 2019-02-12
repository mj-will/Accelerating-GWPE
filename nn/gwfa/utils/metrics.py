
import numpy as np
from gwfa.utils.general import fuzz

def kullback_leibler_divergence(P, Q):
    """
    Compute the Kullback-Leibler divergence for two distributions.
    """
    P = P + fuzz()
    Q = Q + fuzz()
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
