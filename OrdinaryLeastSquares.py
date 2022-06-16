import numpy as np


def ols(x, y):
    # get number of data points
    n = len(x)
    # these are obtained by taking the partial derivative of the SSe with respect
    # to each unknown coefficient (slope and intercept)
    m = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x ** 2) - sum(x) ** 2)
    b = np.mean(y) - m * np.mean(x)
    return m, b
