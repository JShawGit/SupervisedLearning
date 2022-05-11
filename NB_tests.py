import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns; sns.set()
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from keras import datasets
from keras.datasets import cifar10


(X_trainBatch, y_trainBatch), (X_testBatch, y_testBatch) = cifar10.load_data()



labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



X_trainBatch = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_trainBatch])
X_testBatch = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_testBatch])

print()

