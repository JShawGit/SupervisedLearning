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
def read_dataset(data_label):
    import numpy as np
    import pickle
    CLASSES = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9
    }
    train_data = []

    for i in range(1, 6):
        with open("data_batch_" + str(i), 'rb') as fo:  # load CIFAR-10 dataset
            train_data.append(pickle.load(fo, encoding='latin1'))

    with open("test_batch", 'rb') as fo:  # load CIFAR-10 dataset
        test_data = pickle.load(fo, encoding='latin1')

    y_test = np.asarray(test_data['labels']).reshape((10000, 1))
    y_test = np.where(y_test == CLASSES[data_label], 1, 0).T

    x_test = test_data['data'].T

    labels_0 = np.asarray(train_data[0]['labels']).reshape((10000, 1))
    labels_1 = np.asarray(train_data[1]['labels']).reshape((10000, 1))
    labels_2 = np.asarray(train_data[2]['labels']).reshape((10000, 1))
    labels_3 = np.asarray(train_data[3]['labels']).reshape((10000, 1))
    labels_4 = np.asarray(train_data[4]['labels']).reshape((10000, 1))

    y_train = np.concatenate((labels_0, labels_1, labels_2, labels_3, labels_4), axis=0)

    y_train = np.where(y_train == CLASSES[data_label], 1, 0).T

    train_data_0 = train_data[0]['data'].T
    train_data_1 = train_data[1]['data'].T
    train_data_2 = train_data[2]['data'].T
    train_data_3 = train_data[3]['data'].T
    train_data_4 = train_data[4]['data'].T

    x_train = np.concatenate((train_data_0, train_data_1, train_data_2, train_data_3, train_data_4), axis=1)

    # standardize data set
    train_set_x = x_train / 255.
    test_set_x = x_test / 255.

    return train_set_x, test_set_x, y_train, y_test
# --- End Read Dataset --- #
