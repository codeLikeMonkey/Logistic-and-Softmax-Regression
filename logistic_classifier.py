import numpy as np
import load_train_data
import load_test_data
from logistic_regression import *





if __name__ == "__main__":
    images_train_2 = load_train_data.fetch_image(2)
    images_train_3 = load_train_data.fetch_image(3)
    train_input_2 = np.concatenate((images_train_2, np.ones((1, images_train_2.shape[0])).T), axis=1).T
    train_input_3 = np.concatenate((images_train_3, np.ones((1, images_train_3.shape[0])).T), axis=1).T
    train_target = np.concatenate((np.ones((1, train_input_2.shape[1])), np.zeros((1, train_input_3.shape[1]))), axis=1).T
    train_input = np.concatenate((train_input_2, train_input_3), axis=1)

    W,steps = gradient(train_input, train_target)



    images_test_2 = load_test_data.fetch_image(2)
    images_test_3 = load_test_data.fetch_image(3)
    test_input_2 = np.concatenate((images_test_2, np.ones((1, images_test_2.shape[0])).T), axis=1).T
    test_input_3 = np.concatenate((images_test_3, np.ones((1, images_test_3.shape[0])).T), axis=1).T
    test_target = np.concatenate((np.ones((1, test_input_2.shape[1])), np.zeros((1, test_input_3.shape[1]))), axis=1).T
    test_input = np.concatenate((test_input_2, test_input_3), axis=1)

    accuracy = 1 - (sum(sigmoid(W, test_input_2) < 0.5) + sum(sigmoid(W, test_input_3) > 0.5)) / (test_input_2.shape[1] + test_input_3.shape[1])
    print(accuracy)
    print(steps)








