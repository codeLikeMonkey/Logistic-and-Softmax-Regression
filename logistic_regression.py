import numpy as np

def sigmoid(W,X):

    Y = 1 / (1 + np.exp( -np.dot(W.T,X)))

    return Y.T

def gradient(X,T):

    W = np.random.random(X.shape[0]).reshape(X.shape[0],1)
    alpha = 0.001

    # while np.linalg.norm(np.dot((T - sigmoid(W,X)).T,X.T)) > 0.0001:
    #     W = W + alpha * np.dot((T - sigmoid(W,X)).T,X.T)
    #     print(W)
    steps = 0
    while np.linalg.norm(np.dot(X,T-sigmoid(W,X))) > 0.1:
        W = W + alpha * np.dot(X,T-sigmoid(W,X))
        steps = steps + 1
        print("%s-----%s"%(steps,np.linalg.norm(np.dot(X,T-sigmoid(W,X)))))
    return W,steps



if __name__ == "__main__":
    X = np.array([0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50])
    X = np.vstack((X,np.ones(len(X))))
    Y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]).reshape(X.shape[1],1)
    print(gradient(X,Y))
