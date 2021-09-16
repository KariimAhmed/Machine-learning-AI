import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================
    h_theta = sigmoid(np.dot(X, theta))  # hypothesis here differ from linear reg by sigmoid
    theta_reg=theta[1:]
    cost = (np.sum(np.dot(-y,np.log(h_theta)) -np.dot((1-y),np.log(1 - h_theta))) / m )+(lmd/(2*m))*np.sum(theta_reg**2)

    error=(np.dot(X.T,(h_theta - y)) / m).flatten()
    grad[0] = error[0]
    grad[1:] = error[1:]+(lmd/m)*theta_reg
    return cost, grad
