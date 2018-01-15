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


def gradient_descent(Lamada, train_set, train_target):

    train_set_accuracy = []
    train_weights_mat = np.zeros((785,10))
    train_weights_mat[-1,:] = 1
    # hold_out_set = train_set_all[:,0:train_set_all.shape[1]//10]
    # hold_out_target = train_target_all[0:train_set_all.shape[1]//10,:]
    # train_set = train_set_all[:,train_set_all.shape[1]//10:]
    # train_target = train_target_all[train_set_all.shape[1]//10:,:]

    max_epoch = 4
    number_of_mini_batches = 100

    # batches_sizes = np.array([train_set.shape[1]//number_of_mini_batches] * number_of_mini_batches)
    random_index = np.arange(train_set.shape[1]// number_of_mini_batches * number_of_mini_batches)
    np.random.shuffle(random_index)

    random_mini_batches_set  = [train_set[:,x ]for x in np.split(random_index,number_of_mini_batches)]
    random_mini_batches_target = [train_target[x,:]for x in np.split(random_index,number_of_mini_batches)]

    #make hold_out set
    hold_out_set = random_mini_batches_set[-1]
    hold_out_target = random_mini_batches_target[-1]

    #make training set
    random_mini_batches_target = random_mini_batches_target[0:-1]
    random_mini_batches_set = random_mini_batches_set[0:-1]
    hold_out_accuracy = []


    eta_0 = 0.001
    T = 50
    update_times = 0

    for epoch in range(max_epoch):
        for i in range(len(random_mini_batches_set)):
            # print("mini_batch:%s"%i)
            #ith mini_batch_set and mini_batch_target
            update_times += 1
            eta = eta_0 / (1 + update_times / T)
            train_set_accuracy.append(check(train_weights_mat,train_set,train_target))
            hold_out_accuracy.append(check(train_weights_mat,hold_out_set,hold_out_target))
            # print("update times : %s accuracy %s" % (update_times,train_set_accuracy[-1]))
            train_weights_mat = train_weights_mat + eta * (np.dot(random_mini_batches_set[i],random_mini_batches_target[i] - sigmoid(train_weights_mat,random_mini_batches_set[i])) - Lamada * 2 * train_weights_mat)
        if len(train_set_accuracy)>4 and train_set_accuracy[-4]<train_set_accuracy[-3]<train_set_accuracy[-2]<train_set_accuracy[-1]:
            break
        # if len(hold_out_accuracy)>4 and hold_out_accuracy[-4]<hold_out_accuracy[-3]<hold_out_accuracy[-2]<hold_out_accuracy[-1]:
        #     break




    # for update_times in range(500):
    #     eta = eta_0 / (1 + update_times / T)
    #     train_set_accuracy.append(check(train_weights_mat,train_set,train_target))
    #     train_weights_mat = train_weights_mat + eta * (np.dot(train_set, train_target - sigmoid(train_weights_mat, train_set)) - Lamada * 2 * train_weights_mat)

    return train_set_accuracy,hold_out_accuracy


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

    images = images[0:20000]
    labels = labels[0:20000]

    # intialize target maxtrix
    train_target = np.zeros((len(labels),10))
    # mak
    for i in range(len(labels)):
        train_target[i,labels[i]] = 1

    train_images = np.concatenate((np.array(images), np.ones((len(images), 1))), axis=1).T

    return train_images, train_target



if __name__ == "__main__":
    # train_set : 785 * 20000
    # train_target : 20000 * 10
    train_set, train_target = make_train_data()
    # train_weights : 785 * 10
    #
    train_accuracy,hold_out_accuracy = gradient_descent(0.1,train_set,train_target)
    plot_accuracy(train_accuracy)
    plot_accuracy(hold_out_accuracy)
    plt.gca().legend(["train","hold out"])

    plt.show()














