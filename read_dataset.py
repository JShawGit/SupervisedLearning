import numpy
import tensorflow
from tensorflow import keras
""" Get CIFAR 10 --------------------------------------------------------------------------------------------------- """
def getCifar():
    from keras.datasets import cifar10
    return cifar10.load_data()
# --- End Get CIFAR 10 --- #



""" Unzips A File -------------------------------------------------------------------------------------------------- """
def unzip(filename):
    import tarfile
    with tarfile.open(filename) as f:
        print(f.getnames())
        f.extractall('./dataset')
# --- End Unzip --- #



""" Unpickles A File ----------------------------------------------------------------------------------------------- """
def unpickle(filename):
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data
# --- End Unpickle --- #



""" Read Dataset --------------------------------------------------------------------------------------------------- """
def read_dataset(path): #data_label
    import numpy as np

    # get training batches
    train_data = []
    for i in range(1, 6):
        train_data.append(unpickle(path + "data_batch_" + str(i)))

    # get test batch
    test_data = unpickle(path + "test_batch")

    # pre-process test data
    y_test = np.asarray(test_data['labels']).reshape((10000, 1))
    #y_test = np.where(y_test == data_label, 1, 0).T
    x_test = test_data['data'].T

    # reshape training labels
    labels_0 = np.asarray(train_data[0]['labels']).reshape((10000, 1))
    labels_1 = np.asarray(train_data[1]['labels']).reshape((10000, 1))
    labels_2 = np.asarray(train_data[2]['labels']).reshape((10000, 1))
    labels_3 = np.asarray(train_data[3]['labels']).reshape((10000, 1))
    labels_4 = np.asarray(train_data[4]['labels']).reshape((10000, 1))

    # get training labels
    y_train = [labels_0, labels_1, labels_2, labels_3, labels_4]
        #np.concatenate((labels_0, labels_1, labels_2, labels_3, labels_4), axis=0)
    #y_train = np.where(y_train == data_label, 1, 0).T

    # get training data
    train_data_0 = train_data[0]['data'].T
    train_data_1 = train_data[1]['data'].T
    train_data_2 = train_data[2]['data'].T
    train_data_3 = train_data[3]['data'].T
    train_data_4 = train_data[4]['data'].T
    x_train = [train_data_0, train_data_1, train_data_2, train_data_3, train_data_4]
        #= np.concatenate((train_data_0, train_data_1, train_data_2, train_data_3, train_data_4), axis=1)

    # standardize data set
    for i in range(len(x_train)):
        x_train[i] = x_train[i] / 255.
    x_test = x_test / 255.

    return x_train, x_test, y_train, y_test
# --- End Read Dataset --- #
