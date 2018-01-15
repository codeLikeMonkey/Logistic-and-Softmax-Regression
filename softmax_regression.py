import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


def check(W,X,Target):

    A  = np.dot(W.T,X)
    M = A - np.max(A, axis = 0)
    M = np.exp(M)
    M = M / np.sum(M, axis = 0)

    prediction = np.argmax(M, axis = 0)
    accuracy = np.sum(prediction - np.argmax(Target.T, axis=0) == 0) / M.shape[1]
    return accuracy

def sigmoid(W,X):

    z = np.dot(W.T,X)
    Y = 1.0 / (1.0 + np.exp(-z))

    return Y.T


def gradient_descent(train_images, train_target):

    train_set_accuracy = []
    train_weights_mat = np.zeros((785,10))
    train_weights_mat[-1,:] = 1


    eta_0 = 0.001
    T = 100

    for update_times in range(1000):
        eta = eta_0 / (1 + update_times / T)
        train_set_accuracy.append(check(train_weights_mat,train_images,train_target))
        train_weights_mat = train_weights_mat + eta * np.dot(train_images, train_target - sigmoid(train_weights_mat, train_images))

    return train_set_accuracy


def plot_accuracy(accuracy):
    '''

    :param accuracy: accuracy for all epochs, 1 * N vector
    :return: accuracy plot
    '''
    x = np.arange(len(accuracy))
    y = np.array(accuracy) * 100

    # plt.plot(x,y,marker = '.',linestyle = '-')
    plt.plot(x, y)


def make_train_data():
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    images, labels = mndata.load_training()

    images = images[0:2000]
    labels = labels[0:2000]

    # intialize target maxtrix
    train_target = np.zeros((len(labels),10))
    # mak
    for i in range(len(labels)):
        train_target[i,labels[i]] = 1

    train_images = np.concatenate((np.array(images), np.ones((len(images), 1))), axis=1).T

    return train_images, train_target



if __name__ == "__main__":
    # train_images : 785 * 20000
    # train_target : 20000 * 10
    train_images, train_target = make_train_data()
    # train_weights : 785 * 10
    #
    train_accuracy = gradient_descent(train_images,train_target)
    plot_accuracy(train_accuracy)
    plt.show()














