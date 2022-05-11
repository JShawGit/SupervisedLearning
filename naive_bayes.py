import pickle
import numpy as np
from scipy.stats import norm
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from keras import datasets
from keras.datasets import cifar10
import cv2
from sklearn.model_selection import train_test_split
from scipy.stats import norm, multivariate_normal
import math
from skimage.transform import resize

""" Naive Bayes ======================================================================================================
    A supervised learning method.
"""


def read_dataset_easy(num_class, data_label):
    (x_train_batch, y_train_batch), (x_test_batch, y_test_batch) = cifar10.load_data()
    # gray scale images
    # x_train_batch = np.array(x_train_batch)
    # = np.array(x_test_batch)

    x_train_batch = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_train_batch])
    x_test_batch = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_test_batch])

    x_train, x_test, y_train, y_test = train_test_split(x_train_batch, y_train_batch, test_size=0.20, random_state=0)

    if num_class <= 2:
        y_test = np.where(y_test == data_label, 1, 0)
    else:
        y_test_ndex = np.where(y_test >= num_class - 1)
        y_test[y_test_ndex] = num_class - 1

    if num_class <= 2:
        y_train = np.where(y_train == data_label, 1, 0)
    else:
        y_train_ndex = np.where(y_train >= num_class - 1)
        y_train[y_train_ndex] = num_class - 1

    x_train = np.array(x_train).reshape(len(x_train), 1024).astype(float)

    x_test = np.array(x_test).reshape(len(x_test), 1024).astype(float)

    x_test_batch = np.array(x_test_batch).reshape(len(x_test_batch), 1024).astype(float)

    return x_train, x_test, y_train, y_test, x_test_batch, y_test_batch


def fit(x_train, x_test, y_train, y_test, num_class, binary_class):
    mu, sigma, p = naivebayes_learn(x_train, y_train, num_class, binary_class)
    naive_bayes_pred_test = []
    naive_bayes_pred_train = []

    for i in range(x_train.shape[0]):
        if i < x_test.shape[0]:
            naive_bayes_pred_test.append(naivebayes_calssification(x_test[i], mu, sigma, p, num_class, binary_class))
        naive_bayes_pred_train.append(naivebayes_calssification(x_train[i], mu, sigma, p, num_class, binary_class))
    acc_test = classification_accuracy(naive_bayes_pred_test, y_test)
    acc_train = classification_accuracy(naive_bayes_pred_train, y_train)
    print()
    return acc_test, acc_train


def naivebayes_calssification(x_data, mu, sigma, p, num_class, binary_class):
    prob = []
    x = [np.mean(x_data[0:255]),np.mean(x_data[256:511]),np.mean(x_data[512:767]),np.mean(x_data[768:1023])]
    x = [np.mean(x_data)]

    for i in range(num_class):
        # Probabilties for image class

        def cal_pdf(x, mean, stdev):
            exponent = math.exp(-(((x - mean) ** 2) / (2 * stdev ** 2)))
            return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

        prob_temp = 1
        for j in range(len(x)):
            prob_temp *= cal_pdf(x[j], mu[i][j], sigma[i][j])

        # prob.append(num / denom)
        prob.append(prob_temp * p[i])

    max_label = np.argmax(prob)

    return max_label


def classification_accuracy(pred, gt):
    class_count = len(pred)
    class_err = 0
    for index in range(0, class_count):
        if pred[index] != gt[index]:
            class_err += 1

    accuracy_err = class_err / class_count
    accuracy = (1.0 - accuracy_err) * 100
    print("Classification accuracy for the given set: {:3f}".format(accuracy))
    return accuracy


def naivebayes_learn(x_train, y_train, num_class, binary_class):
    mu = np.zeros((num_class, 1))
    sigma = np.zeros((num_class, 1))
    p = np.zeros((num_class, 1))

    for i in range(num_class):
        # store images per loop
        images = []

        for j in range(x_train.shape[0]):
            # Store the sample to this class if label match
            if y_train[j] == i:
                images.append(x_train[j])
        images = np.asarray(images)

        # Calculate mean, standard deviation
        if images.shape[0] != 0:
            mean_temp = []

            mean_temp = np.mean(np.mean(images, axis=0), axis=0)
            #mean_temp.append(np.mean(np.mean(images[:, 0:255], axis=0), axis=0))
            #mean_temp.append(np.mean(np.mean(images[:, 256:511], axis=0), axis=0))
            #mean_temp.append(np.mean(np.mean(images[:, 512:767], axis=0), axis=0))
            #mean_temp.append(np.mean(np.mean(images[:, 768:1023], axis=0), axis=0))

            sigma_temp = []

            sigma_temp = np.mean(np.std(images, axis=0), axis=0)
            #sigma_temp.append(np.mean(np.std(images[:, 0:255], axis=0), axis=0))
            #sigma_temp.append(np.mean(np.std(images[:, 256:511], axis=0), axis=0))
            #sigma_temp.append(np.mean(np.std(images[:, 512:767], axis=0), axis=0))
            #sigma_temp.append(np.mean(np.std(images[:, 768:1023], axis=0), axis=0))

            mu[i] = mean_temp
            sigma[i] = sigma_temp
            p[i] = images.shape[0] / x_train.shape[0]

    return mu, sigma, p


def BAYES(x_train, x_test, y_train, y_test):
    dtc = GaussianNB()
    dtc.fit(x_train, y_train)

    # Accuracy Training
    x_train_pred = dtc.predict(x_train)
    acc_train = metrics.accuracy_score(y_train, x_train_pred)
    print('Accuracy Sklearn Train:', acc_train * 100)

    # Accuracy Testing
    x_test_pred = dtc.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, x_test_pred)
    print('Accuracy Sklearn Test:', acc_test * 100)

    return acc_test * 100, acc_train * 100


class naive_bayes:

    def __init__(self, dim, dataset_label, binary_class, num_class):
        self.dataset_label = dataset_label

        self.x_train, self.x_test, self.y_train, self.y_test, \
        self.x_test_batch, self.y_test_batch \
            = read_dataset_easy(num_class, dataset_label)

        # = read_dataset(dataset_label, batch_num, num_class)
        self.dim = dim
        self.num_class = num_class
        self.binary_class = binary_class

    def naive_bayes_run(self):
        acc_test, acc_train = fit(self.x_train, self.x_test, self.y_train, self.y_test, self.num_class,
                                  self.binary_class)

        acc_test_skl, acc_train_skl = BAYES(self.x_train, self.x_test, np.ravel(self.y_train), np.ravel(self.y_test))

        return [acc_test, acc_train, acc_test_skl, acc_train_skl]
