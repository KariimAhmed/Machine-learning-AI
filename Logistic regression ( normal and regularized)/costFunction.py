import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================
    h_theta=sigmoid(np.dot(X, theta)) # hypothesis here differ from linear reg by sigmoid
    cost = np.sum(np.dot(-y, np.log(h_theta)) - np.dot(1-y,np.log(1-h_theta)))/m
    grad=np.dot((h_theta-y),X)/m # summation in exercise is not correct

    return cost, grad
