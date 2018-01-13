import numpy as np


def check(W,X,Target):
    return np.sum(np.abs(np.round(sigmoid(W,X)) - Target))/Target.shape[0]

def sigmoid(W,X):
    z = np.dot(W.T,X)
    Y = 1.0 / (1.0 + np.exp(-z))

    return Y.T

def gradient(X,Target):

    W = np.zeros((X.shape[0],1))
    accuracy = []
    eta_0 = 0.0001
    T = 5
    for epoch in range(450):
        eta = eta_0 / (1 + epoch / T)
        W = W + eta * np.dot(X, Target -sigmoid(W, X))
        # print(1-check(W,X,Target))
        accuracy.append(1-check(W,X,Target))
    return W,accuracy



if __name__ == "__main__":
    X = np.array([0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50])
    X = np.vstack((X,np.ones(len(X))))
    Y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]).reshape(X.shape[1],1)
    W = gradient(X,Y)
    print(check(W,X,Y))
    print(np.round(sigmoid(W,X)))
