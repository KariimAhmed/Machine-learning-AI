import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)
    for i in range(0, num_iters):
       #theta=theta-alpha*(1/m)*np.sum((np.dot(X[i],theta)-y[i])**2)

       error=np.dot(X,theta).flatten()-y
       theta -= (alpha/m)*np.sum(X*error[:, np.newaxis], 0)   #np.newaxis increase the dimension
       J_history[i] = compute_cost(X, y, theta) #save the cost for every iteration
       #theta_0[i] = theta[0]
       #theta_1[i] = theta[1]
    return theta, J_history#,theta_0#,theta_1


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # ===========================================================
        # Save the cost every iteration
        error = np.dot(X, theta).flatten() - y
        theta -= (alpha / m) * np.sum(X * error[:, np.newaxis], 0)  # np.newaxis increase the dimension
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
    