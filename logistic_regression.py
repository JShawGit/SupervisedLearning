import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

""" Logistic Regression ======================================================================================================
    A supervised learning method.
"""
def read_dataset(data_label):
    train_data = []

    for i in range(1, 6):
        with open("data_batch_" + str(i), 'rb') as fo:  # load CIFAR-10 dataset
            train_data.append(pickle.load(fo, encoding='bytes'))

    with open("test_batch", 'rb') as fo:  # load CIFAR-10 dataset
        test_data = pickle.load(fo, encoding='bytes')

    y_test = np.asarray(test_data[b'labels']).reshape((10000, 1))
    y_test = np.where(y_test == data_label, 1, 0).T

    x_test = test_data[b'data'].T

    labels_0 = np.asarray(train_data[0][b'labels']).reshape((10000, 1))
    labels_1 = np.asarray(train_data[1][b'labels']).reshape((10000, 1))
    labels_2 = np.asarray(train_data[2][b'labels']).reshape((10000, 1))
    labels_3 = np.asarray(train_data[3][b'labels']).reshape((10000, 1))
    labels_4 = np.asarray(train_data[4][b'labels']).reshape((10000, 1))

    y_train = np.concatenate((labels_0, labels_1, labels_2, labels_3, labels_4), axis=0)

    y_train = np.where(y_train == data_label, 1, 0).T

    train_data_0 = train_data[0][b'data'].T
    train_data_1 = train_data[1][b'data'].T
    train_data_2 = train_data[2][b'data'].T
    train_data_3 = train_data[3][b'data'].T
    train_data_4 = train_data[4][b'data'].T

    x_train = np.concatenate((train_data_0, train_data_1, train_data_2, train_data_3, train_data_4), axis=1)

    # standardize data set
    train_set_x = x_train / 255.
    test_set_x = x_test / 255.

    return train_set_x, test_set_x, y_train, y_test


# Sigmoid function
def sigmoid(z):
    sig = 1 / (1 + np.exp(-1 * z))

    return sig


# initialize and zero w and b
def initialize_zeros(dim_param):
    w = np.zeros((dim_param, 1))
    b = 0

    assert (w.shape == (dim_param, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# Calculating gradients and costs
def calc_cost_grads(w, b, x, y):
    m = x.shape[1]
    sigA = sigmoid(np.dot(w.T, x) + b)

    cost = np.sum(y * np.log(sigA) + ((1 - y) * np.log(1 - sigA))) / -m

    dw = np.dot(x, (sigA - y).T) / m

    db = (np.sum(sigA - y)) / m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)

    assert (cost.shape == ())

    grads = {'dw': dw,
             'db': db
             }

    return grads, cost


# Optimizing weights and bias
def train_lr(w, b, x, y, num_iterations, learning_rate, print_flag, step):
    costs = []

    for i in range(num_iterations):
        grads, cost = calc_cost_grads(w, b, x, y)

        dw = grads['dw']
        db = grads['db']

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % step == 0:
            costs.append(cost)

        if print_flag and i % step == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {'w': w,
              'b': b
              }

    grads = {'dw': dw,
             'db': db
             }

    return params, grads, costs


# Prediction method
def predict(w, b, x_data):
    m = x_data.shape[1]
    y = np.zeros((1, m))
    w = w.reshape(x_data.shape[0], 1)
    sigA = sigmoid(np.dot(w.T, x_data) + b)

    for i in range(sigA.shape[1]):

        if sigA[0, i] < 0.5:
            y[0, i] = 0
        elif sigA[0, i] > 0.5:
            y[0, i] = 1
        pass

    assert (y.shape == (1, m))

    return y


class logistic_regression:
    """ Init Tree -------------------------------------------------------------------------------------------------- """

    def __init__(self, iterations, lr, cost_flag, dim, dataset_label):
        self.num_iterations = iterations
        self.learning_rate = lr
        self.print_cost = cost_flag
        self.dataset_label = dataset_label
        self.x_train, self.x_test, self.y_train, self.y_test \
            = read_dataset(dataset_label)

        self.dim = dim

    # --- End Init logistic Regression --- #

    # Logistic Regression driver
    def log_reg_model(self, step):
        w, b = initialize_zeros(self.dim)

        params, grads, costs = train_lr(w, b, self.x_train,
                                        self.y_train, self.num_iterations,
                                        self.learning_rate, self.print_cost,
                                        step)

        w = params['w']
        b = params['b']

        y_predict_train = predict(w, b, self.x_train)
        y_predict_test = predict(w, b, self.x_test)

        model_data = {
            'costs': costs,
            'y_predict_train': y_predict_train,
            'y_predict_test': y_predict_test,
            'w': w,
            'b': b,
            'learning_rate': self.learning_rate,
            'num_iterations': self.num_iterations,
            'train_acc': 100.0 - np.mean(np.abs(y_predict_train - self.y_train)) * 100.0,
            'test_acc':  100.0 - np.mean(np.abs(y_predict_test  - self.y_test))  * 100.0
        }

        print("train accuracy: {} %".format(model_data['train_acc']))
        print("test accuracy:  {} %".format(model_data['test_acc']))

        return model_data
