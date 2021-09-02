import matplotlib.pyplot as plt
import numpy as np
from gradientDescent import *
from plotData import *
from featureNormalize import *
from normalEqn import *
#plot data
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
X=data[:,0:2]
y=data[:,2]
m=y.size
plt.ion() # turn on plotting interactive mode
plt.figure(0) #create figure to plot into it
#Plot 1 (Price vs size)
plt.scatter(X[:,0],y,marker='x', color='r')
plt.ylabel('House Prices')
plt.xlabel('Size of house in ft^2')
plt.show()
#Plot 2 (Price vs no. of bedrooms)
plt.figure(2)
plt.scatter(X[:,1],y,marker='x', color='b')
plt.ylabel('House Prices')
plt.xlabel('No. of bedrooms')
plt.show()
X_un=X # un normalized X
X_un = np.c_[np.ones(m), X_un]
X, mu, sigma=feature_normalize(X) # V.I normailze X
X = np.c_[np.ones(m), X]  # Add a column of ones to X

# Choose some alpha value
alpha = 0.45# IF WE HAVE nan REDUCE alpha
num_iters = 50 # also inc or dec to converge
# Init theta and Run Gradient Descent
theta = np.zeros(3) # 3 theta bec 2 features so theta0,theta1,theta2
theta1, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)
J1=J_history

#try different values for alpha
alpha = 0.05
num_iters = 50
theta = np.zeros(3)
theta2, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)
J2=J_history

#try different values for alpha
alpha = 0.01
num_iters = 50
theta = np.zeros(3)
theta3, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)
J3=J_history


# Plot the convergence graph
plt.figure(3)
plt.plot(np.arange(J_history.size), J1,c='r',label='J1=0.1') # NO. OF ITERATIONS VS CONST FUNCTION
plt.plot(np.arange(J_history.size), J2,c='b',label='J2=0.05')
plt.plot(np.arange(J_history.size), J3,c='y',label='J3=0.01')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
# Display gradient descent's result
print('Theta computed from gradient descent : \n{}'.format(theta))


# we will continue with J1 (the best convergence)

#prediction
# Predict values for a house with 1650 square feet and
# 3 bedrooms.
#h=theta0*X0+theta1*X1+theta2*X2 so its a dot product
# in this example X1 is the space and X2 no of bedrooms and X0=1
#since theta is obtained by normalizing the features , so at prediction we have to
# normalize X1 and X2 but Xo is kept the same because it's all ones
price= 0
features= np.array([1650, 3])
features= (features-mu)/sigma
features= np.r_[(1,features)] #(add one to the array) translates slice objects to concatenation along the first axis.
price = np.dot(features, theta1)
print('For house of area = 1650 ft^2 and 3 bedrooms , it costs about',price)

# Normal euqation ( the exact solution if data is linear)
# obtain theta using normal equation with no need to feature normalization and without
# gradient descent ( works only for linear features)
X = X_un # X agai will be the un normalized
theta=normal_eqn(X, y)
price= 0
features1= np.array([1,1650, 3])
price = np.dot(features1, theta)
print('For house of area = 1650 ft^2 and 3 bedrooms , it costs using normal equation about',price)
# so here the two prices are nearly the same after changing alpha to 0.45
