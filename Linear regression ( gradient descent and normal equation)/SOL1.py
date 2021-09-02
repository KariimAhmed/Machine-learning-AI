del()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from computeCost import *
from gradientDescent import *
from plotData import *
#plot data
print('Plotting Data...') #print this text
data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))
X=data[:,0]
y=data[:,1]
m=y.size
plt.ion() # turn on plotting interactive mode
plt.figure(0) #create figure to plot into it
#plot_data(X,y)
#plt.plot(X,y)
plt.scatter(X,y,marker='x', color='r')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

##linear regression
#batch gradient descent
X=np.c_[np.ones(m),X]# add a coloumn of one to x to accomodate for theta0
theta=np.zeros(2)    #Intialize theta
#gradient descent parameters
iterations=1500
alpha=0.01
print(compute_cost(X,y,theta))
theta, J_history= gradient_descent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: ' + str(theta.reshape(2)))
# Plot the linear fit
plt.figure(0)
plt.scatter(X[:,1],y,marker='x', color='r')
plt.plot(X[:,1],np.dot(X,theta))
plt.show()

#prediction
# Predict values for population sizes of 35,000 and 70,000
#h=theta0*X0+theta1*X1 , so its a dot product
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))

theta_0=np.linspace(-10,10,100) #here we have 2 theta theta0 and theta1 bec we have only one input
theta_1=np.linspace(-1,4,100)
#initialize J vals to a matrix of 0's
xs,ys=np.meshgrid(theta_0,theta_1)
J_vals = np.zeros(xs.shape)
#Fill out J vals
for i in range(0, theta_0.size):
    for j in range(0, theta_1.size):
       t = np.array([theta_0[i],theta_1[j]])
       J_vals[i][j] = compute_cost(X, y, t)
J_vals=np.transpose(J_vals)

# plot theta 1 and theta 2
fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals,cmap=cm.coolwarm)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()

# contours of J and Theta
fig2 = plt.figure(2)
plt.contour(xs, ys, J_vals,levels=np.logspace(-2,3,20))
plt.plot(theta[0], theta[1], c='r', marker="x")
plt.show()