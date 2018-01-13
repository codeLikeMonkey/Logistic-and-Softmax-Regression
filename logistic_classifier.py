import numpy as np
import load_train_data
import load_test_data
from logistic_regression import *
import matplotlib.pyplot as plt

def plot_weights(Weights):
    plt.imshow(Weights[0:-1].reshape(28,28))
    plt.show()

def plot_accuracy(accuracy):
    x = np.arange(len(accuracy))
    y = np.array(accuracy)

    plt.plot(x,y)
    plt.show()


def make_train_data(numberA,numberB):
    train_image_A = load_train_data.fetch_image(numberA)
    train_image_B = load_train_data.fetch_image(numberB)
    train_input_A = np.concatenate((train_image_A, np.ones((1, train_image_A.shape[0])).T), axis=1).T
    train_input_B = np.concatenate((train_image_B, np.ones((1, train_image_B.shape[0])).T), axis=1).T
    train_target = np.concatenate((np.ones((1, train_input_A.shape[1])), np.zeros((1, train_input_B.shape[1]))), axis=1).T
    train_input = np.concatenate((train_input_A, train_input_B), axis=1)


    #make hold out set



    return train_input,train_target

def make_test_data(numberA,numberB):
    test_image_A = load_test_data.fetch_image(numberA)
    test_image_B = load_test_data.fetch_image(numberB)
    test_input_A = np.concatenate((test_image_A, np.ones((1, test_image_A.shape[0])).T), axis=1).T
    test_input_B = np.concatenate((test_image_B, np.ones((1, test_image_B.shape[0])).T), axis=1).T
    test_target = np.concatenate((np.ones((1, test_input_A.shape[1])), np.zeros((1, test_input_B.shape[1]))), axis=1).T
    test_input = np.concatenate((test_input_A, test_input_B), axis=1)

    return test_input,test_target


if __name__ == "__main__":

    train_input_2_vs_3,train_target_2_vs_3 = make_train_data(2,3)

    Weights_2_vs_3,accuracy_2_vs_3 = gradient(train_input_2_vs_3, train_target_2_vs_3)

    test_input_2_vs_3,test_target_2_vs_3 = make_test_data(2,3)

    plot_weights(Weights_2_vs_3)
    plot_accuracy(accuracy_2_vs_3)








