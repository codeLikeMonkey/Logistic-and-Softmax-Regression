from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np

mndata = MNIST('./mnist_data')
mndata.gz = True
images, labels = mndata.load_training()

def fetch_image(number):
    new_labels = np.array(list(labels))
    indexs = np.arange(len(new_labels))
    image_select = indexs[new_labels == number]
    return np.array(images)[image_select]



if __name__ == "__main__":
    images_2 = fetch_image(3)
    plt.imshow(images_2[2].reshape([28,28]))
    plt.show()