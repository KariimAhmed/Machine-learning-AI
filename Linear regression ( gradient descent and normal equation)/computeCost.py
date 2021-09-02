import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    cost=np.sum((np.dot(X,theta)-y)**2)/(2*m)
   # cost_m = np.sum((np.dot(X, theta) - y) ** 2) / (2 * m)
    return cost
