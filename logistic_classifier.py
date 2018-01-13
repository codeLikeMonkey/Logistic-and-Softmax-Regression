import numpy as np
import load_train_data
import load_test_data
# from logistic_regression import *
import matplotlib.pyplot as plt


def check(W,X,Target):
    return np.sum(np.abs(np.round(sigmoid(W,X)) - Target))/Target.shape[0]

def sigmoid(W,X):
    z = np.dot(W.T,X)
    Y = 1.0 / (1.0 + np.exp(-z))

    return Y.T

def loss(Target,Y):
    E = 0
    # for n in range(len(Target)):
        # E = E + np.power(Target[n],Y[n]) * np.power(1-Y[n],Target[n])
    E = np.dot(np.log10(Y.T+1e-3),Target) + np.dot(np.log10(1 - Y.T+1e-3),1 - Target)
    # print()
    return -float(E)


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


def early_stoping(train_set,train_target,hold_out_set,hold_out_target,test_set,test_target):
    
    train_set_weights = np.zeros((train_set.shape[0],1))
    train_set_weights[-1] = 1
    train_set_accuracy = []
    hold_out_accuracy = []
    test_set_accuracy = []
    train_set_loss = []
    hold_out_loss = []
    test_set_loss = []
    eta_0 = 0.001
    T = 10
    
    for epoch in range(450):

        train_set_accuracy.append(1 - check(train_set_weights,train_set,train_target))
        hold_out_accuracy.append(1 - check(train_set_weights,hold_out_set,hold_out_target))
        test_set_accuracy.append(1 - check(train_set_weights,test_set,test_target))
        #
        train_set_loss.append(loss(train_target,sigmoid(train_set_weights,train_set)))
        hold_out_loss.append(loss(hold_out_target,sigmoid(train_set_weights,hold_out_set)))
        test_set_loss.append(loss(test_target,sigmoid(train_set_weights,test_set)))

        eta = eta_0 / (1 + epoch / T)
        train_set_weights = train_set_weights + eta * np.dot(train_set,train_target - sigmoid(train_set_weights,train_set))
    
    return train_set_weights,train_set_accuracy,hold_out_accuracy,test_set_accuracy,train_set_loss,hold_out_loss,test_set_loss
        
    

def plot_weights(Weights):
    plt.imshow(Weights[0:-1].reshape(28,28))

def plot_accuracy(accuracy):
    x = np.arange(len(accuracy))
    y = np.array(accuracy) * 100

    # plt.plot(x,y,marker = '.',linestyle = '-')
    plt.plot(x, y)
def plot_loss(loss,color):
    x = np.arange(len(loss))
    y = np.array(loss)

    # plt.plot(x, y, marker='.', linestyle='-',color = color)
    plt.plot(x, y)




def make_train_data(numberA,numberB):

    image_A = load_train_data.fetch_image(numberA)
    image_B = load_train_data.fetch_image(numberB)

    #make_train_set

    train_image_A = image_A[np.fix(image_A.shape[0] * 0.1).astype(int):]
    train_image_B = image_B[np.fix(image_B.shape[0] * 0.1).astype(int):]
    train_input_A = np.concatenate((train_image_A, np.ones((1, train_image_A.shape[0])).T), axis=1).T
    train_input_B = np.concatenate((train_image_B, np.ones((1, train_image_B.shape[0])).T), axis=1).T
    train_target = np.concatenate((np.ones((1, train_input_A.shape[1])), np.zeros((1, train_input_B.shape[1]))), axis=1).T
    train_input = np.concatenate((train_input_A, train_input_B), axis=1)

    #make hold-out set
    hold_out_image_A = image_A[:np.fix(image_A.shape[0] * 0.1).astype(int)]
    hold_out_image_B = image_B[:np.fix(image_B.shape[0] * 0.1).astype(int)]

    hold_out_input_A = np.concatenate((hold_out_image_A, np.ones((1, hold_out_image_A.shape[0])).T), axis=1).T
    hold_out_input_B = np.concatenate((hold_out_image_B, np.ones((1, hold_out_image_B.shape[0])).T), axis=1).T
    hold_out_target = np.concatenate((np.ones((1, hold_out_input_A.shape[1])), np.zeros((1, hold_out_input_B.shape[1]))), axis=1).T
    hold_out_input = np.concatenate((hold_out_input_A, hold_out_input_B), axis=1)


    return train_input,train_target,hold_out_input,hold_out_target

def make_test_data(numberA,numberB):
    test_image_A = load_test_data.fetch_image(numberA)
    test_image_B = load_test_data.fetch_image(numberB)
    test_input_A = np.concatenate((test_image_A, np.ones((1, test_image_A.shape[0])).T), axis=1).T
    test_input_B = np.concatenate((test_image_B, np.ones((1, test_image_B.shape[0])).T), axis=1).T
    test_target = np.concatenate((np.ones((1, test_input_A.shape[1])), np.zeros((1, test_input_B.shape[1]))), axis=1).T
    test_input = np.concatenate((test_input_A, test_input_B), axis=1)

    return test_input,test_target




if __name__ == "__main__":


    # 2 vs 3

    train_input_2vs3,train_target_2vs3,hold_out_input_2vs3,hold_out_target_2vs3 = make_train_data(2,3)
    test_input_2vs3, test_target_2vs3 = make_test_data(2, 3)

    train_Weights_2vs3,train_accuracy_2vs3,hold_out_accuracy_2vs3,test_set_accuracy_2vs3,train_set_loss_2vs3,hold_out_loss_2vs3,test_set_loss_2vs3 = early_stoping(train_input_2vs3, train_target_2vs3,hold_out_input_2vs3,hold_out_target_2vs3,test_input_2vs3,test_target_2vs3)


    #train_Weights_2vs3,train_accuracy_2vs3 = gradient(train_input_2vs3,train_target_2vs3)

    plt.figure(1)
    ax = plt.gca()
    ax.set_title("Weights Pattern 2 vs 3")
    plot_weights(train_Weights_2vs3)
    plt.show()
    #
    plt.figure(2)
    ax = plt.gca()
    ax.set_xlim([0,500])
    ax.set_title("Correction Rate over Training")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Correction Rate Percentage ")
    plot_accuracy(train_accuracy_2vs3)
    plot_accuracy(hold_out_accuracy_2vs3)
    plot_accuracy(test_set_accuracy_2vs3)
    plt.show()

    plt.figure(3)
    ax = plt.gca()
    # ax.grid()
    ax.set_title("Loss (E) over Training")
    ax.set_xlabel("Training Epochs")
    plot_loss(train_set_loss_2vs3,'r')
    plot_loss(hold_out_loss_2vs3,'b')
    plot_loss(test_set_loss_2vs3,'g')
    ax.legend(['Training Set','Hold Out Set','Test Set'])
    plt.show()
    # print(train_set_loss_2vs3)










