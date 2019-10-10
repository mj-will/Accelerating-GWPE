
import numpy as np

def ELU(x, alpha=0.01):
    """Exponetial Linear Unit"""
    y = x.copy()
    neg_indices = np.where(x <= 0.)
    y[neg_indices] = alpha * (np.exp(y[neg_indices]) - 1.)
    return y

def IELU(x, alpha=0.01):
    """Inverse of the Exponential Linear Unit"""
    y = x.copy()
    neg_indices = np.where(x <= 0.)
    y[neg_indices] = np.log(y[neg_indices] / alpha + 1.)
    return y

