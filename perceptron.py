from multiprocessing.pool import ThreadPool as Pool
import multiprocessing
import numpy as np

""" Perceptron =========================================================================================================
    A supervised learning method.
"""
class Perceptron:
    """ Init Perceptron -------------------------------------------------------------------------------------------- """
    def __init__(self, num_features, num_classes):
        self.classes = range(num_classes)
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = None
        self.bias = 0
    # --- End Init Perceptron --- #


    """ Reset Weights ---------------------------------------------------------------------------------------------- """
    def reset_weights(self):
        self.weights = np.zeros((self.num_classes, self.num_features))
        self.bias = 0
    # --- End Reset Weights --- #


    """ Predict ---------------------------------------------------------------------------------------------------- """
    def predict(self, X):
        res = []  # X is an array
        for row in range(len(X)):
            arg_max    = None
            prediction = None

            for c in self.classes:
                w = self.weights[c]
                x = X[row]

                # check if the activation is better
                activation = np.dot(np.array(x), w)
                if arg_max is None or activation >= arg_max:
                    arg_max = activation
                    prediction = c

            res.append(prediction)
        return res
    # --- End Predict --- #


    """ Fit -------------------------------------------------------------------------------------------------------- """
    def fit(self, X, Y, epochs=10):
        # prepare to learn
        self.reset_weights()
        num_rows = len(Y)

        # iterate I times for each item in the dataset
        for i in range(epochs):
            for row in range(num_rows):
                x = X[row]
                y = Y[row]

                arg_max    = None
                prediction = None
                # check if the activation is better
                for c in self.classes:
                    w = self.weights[c]
                    activation = np.dot(np.array(x), w)
                    if arg_max is None or activation >= arg_max:
                        arg_max = activation
                        prediction = c

                # update weights if not correct
                if prediction != y:
                    self.weights[y] += x
                    self.weights[prediction] -= x

""" END PERCEPTRON ///////////////////////////////////////////////////////////////////////////////////////////////// """
