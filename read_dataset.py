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
        data = pickle.load(f)
    return data
# --- End Unpickle --- #
