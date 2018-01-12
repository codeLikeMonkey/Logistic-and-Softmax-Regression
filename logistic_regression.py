import numpy as np

def sigmoid(W,X):

    Y = 1 / (1 + np.exp( -np.dot(W.T,X)))

    return Y

def gradient(W,X,Y):

    alpha = 0.03

    return W + alpha * np.dot((Y - sigmoid(W,X)).T,X.T)



if __name__ == "__main__":
    X = np.array([0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50])
    X = np.vstack((X,np.ones(len(X))))
    W = np.random.random(2).T
    Y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]).T
    while np.linalg.norm(W - gradient(W, X, Y)) > 0.0001:
        W = gradient(W, X, Y)
    print(W)
