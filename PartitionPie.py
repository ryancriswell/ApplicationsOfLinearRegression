from GradientDescent import *
from OrdinaryLeastSquares import *


# Partition of the ols sum of squares:
def partition_ols(x, y):
    mean_y = np.mean(y)
    m, b = ols(x, y)
    y_pred = m * x + b
    # error sum of squares (also called residual SS)
    SSe = sum((y - y_pred) ** 2)
    # regression sum of squares (also called explained SS)
    SSr = sum((y_pred - mean_y) ** 2)
    # total sum of squares (sum of sample variance = absolute distance from point to y bar)
    SSt = sum((y - mean_y) ** 2)
    return SSe, SSr


# Partition of the gradient descent sum of squares:
def partition_gd(x, y):
    mean_y = np.mean(y)
    m, b, epoch = gradient_descent(x, y)
    y_pred = m * x + b
    # error sum of squares (also called residual SS)
    SSe = sum((y - y_pred) ** 2)
    # regression sum of squares (also called explained SS)
    SSr = sum((y_pred - mean_y) ** 2)
    # total sum of squares (sum of sample variance = absolute distance from point to y bar)
    SSt = sum((y - mean_y) ** 2)
    return SSe, SSr
