import numpy as np

def normal_eqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    # ===================== Your Code Here =====================
    # Instructions : Complete the code to compute the closed form solution
    #                to linear regression and put the result in theta
    #
   #theta=np.dot(np.linalg.inv(np.matmul(X_un.transpose(),X_un)),np.dot(X_un.transpose(),y))
    Xt=np.transpose(X)
    a1=np.linalg.inv(np.dot(Xt,X)) # inverse(XT*X)
    a2=np.dot(Xt,y)  # XT*Y
    theta=np.dot(a1,a2)
    return theta
