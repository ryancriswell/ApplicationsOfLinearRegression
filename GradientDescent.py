import matplotlib.pyplot as plt
import numpy as np

weights = []
biases = []
costs = []
L = 0.01
max_epochs=1000
threshold = 1e-7

def gradient_descent(x, y):
    # number of data points
    n = len(x)
    # start with m and c at 0
    m = 0
    c = 0
    epoch = 0
    for i in range(max_epochs):
        y_pred = m * x + c
        # partial derivative of the cost (also called loss or error) function with respect to m
        Dm = (-2 / n) * sum(x * (y - y_pred))
        # partial derivative of the cost function with respect to c
        Dc = (-2 / n) * sum(y - y_pred)
        # step size = learning rate * old slope
        # update m: L * Dm is the step size
        m = m - L * Dm
        # update the intercept with the old intercept - step size
        c = c - L * Dc

        # get costs, weights, and biases for plotting
        current_cost = np.sum((y-y_pred)**2) / len(y)
        
        weights.append(m)
        biases.append(c)
        costs.append(current_cost)

        epoch = i
        # check if our step size have fallen below our minimum threshold
        if abs(L * Dm) < threshold and abs(L * Dc) < threshold:
            break

    return m, c, epoch



