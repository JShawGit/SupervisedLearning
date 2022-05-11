from multiprocessing.pool import ThreadPool as Pool
import multiprocessing
import numpy as np

""" Decision Tree ======================================================================================================
    A supervised learning method.
    
    Good References:
    https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf
    https://www.softwaretestinghelp.com/decision-tree-algorithm-examples-data-mining/
    https://github.com/balgot/ib031-cifar10/blob/master/src/CIFAR10.ipynb
"""
class DecisionTree:
    """ Init Tree -------------------------------------------------------------------------------------------------- """
    def __init__(self, min_split=2, max_depth=10):
        self.min_split  = min_split
        self.max_depth  = max_depth
        self.root       = None

        self.n_nodes = 0
        self.leaves  = 0
    # --- End Init Tree --- #


    """ Predict ---------------------------------------------------------------------------------------------------- """
    def predict(self, X):
        if len(X.shape) == 2:
            res = []
            for x in X:
                res.append(self.search_node(x, self.root))
        else:
            res = self.search_node(X, self.root)
        return res
    # --- End Predict --- #


    """ Search Node ------------------------------------------------------------------------------------------------ """
    def search_node(self, X, node):

        # if leaf
        if node.label is not None:
            return node.label

        # if decision node, search left or right
        if X[node.index] > node.condition:
            return self.search_node(X, node.right_branch)
        else:
            return self.search_node(X, node.left_branch)
    # --- End Search Node --- #


    """ Fit -------------------------------------------------------------------------------------------------------- """
    def fit(self, X, Y):
        self.root = self.create_tree(X, Y, 0)
        return
    # --- End Fit --- #


    """ Create Tree ------------------------------------------------------------------------------------------------ """
    def create_tree(self, X, Y, depth):
        samples, features = np.shape(X)

        # split until data is totally separated according to info gains
        if samples >= self.min_split and depth <= self.max_depth:

            # check if the classes can be further split apart, if so create a decision node
            index, threshold, data_R_X, data_R_Y, data_L_X, data_L_Y, gain = self.best_split(X, Y, features)
            print("GAIN: " + str(gain) + "\n")
            if gain > 0:
                R = self.create_tree(data_R_X, data_R_Y, depth + 1)
                L = self.create_tree(data_L_X, data_L_Y, depth + 1)
                self.n_nodes += 1
                return Node(
                    right_branch=R,
                    left_branch=L,
                    index=index,
                    condition=threshold,
                    gain=gain
                )

        # create leaf if data cannot be split
        l = list(Y)
        self.leaves += 1
        return Node(label=max(l, key=l.count))
    # --- End Create Tree --- #


    """ Best Split ------------------------------------------------------------------------------------------------- """
    def best_split(self, X, Y, features):
        index = threshold = data_R_X = data_R_Y = data_L_X = data_L_Y = gain = 0
        max_gain = -float("inf")
        data = np.concatenate((X, np.array(Y).reshape(-1, 1)), axis=1)

        # check each pixel (feature)
        print("Finding the best split...")
        for i in range(features):
            if i % 1000 == 0:
                print("I = " + str(i))
            values = X[:, index]
            thresholds = np.unique(values)

            # check all the values to check for possible splits
            for t in thresholds:
                R = np.array([row for row in data if row[i] > t])
                L = np.array([row for row in data if row[i] <= t])
                if len(R) > 0 and len(L) > 0:
                    R_Y = R[:, -1]
                    L_Y = L[:, -1]
                    g = self.info_gain(Y, R_Y, L_Y)

                    # reset split values
                    if g > max_gain:
                        index     = i
                        threshold = t
                        data_R_X  = R[:, :-1]
                        data_R_Y  = R_Y
                        data_L_X  = L[:, :-1]
                        data_L_Y  = L_Y
                        gain      = g

        return index, threshold, data_R_X, data_R_Y, data_L_X, data_L_Y, gain
    # --- End Best Split --- #


    """ Info Gain -------------------------------------------------------------------------------------------------- """
    def info_gain(self, Y, R_Y, L_Y):
        R_w = len(R_Y) / len(Y)
        L_w = len(L_Y) / len(Y)
        Y_gini = self.gini(Y)
        R_Y_gini = self.gini(R_Y)
        L_Y_gini = self.gini(L_Y)
        return Y_gini - (R_w * R_Y_gini + L_w * L_Y_gini)
    # --- End Info Gain --- #


    """ Gini ------------------------------------------------------------------------------------------------------- """
    def gini(self, Y):
        labels = np.unique(Y)
        gini = 1

        # subtract all of the probabilities/ratios
        for l in labels:
            Y_l = Y[Y == l]
            gini -= (len(Y_l) / len(Y)) ** 2

        return gini
    # --- End Gini --- #
""" END DECISION TREE ////////////////////////////////////////////////////////////////////////////////////////////// """



""" Node ===============================================================================================================
    A node within the tree structure.
"""
class Node:
    """ Init Node -------------------------------------------------------------------------------------------------- """
    def __init__(self, right_branch=None, left_branch=None, index=-1, condition=None, gain=0, label=None):
        self.right_branch   = right_branch
        self.left_branch    = left_branch
        self.index          = index
        self.condition      = condition
        self.gain           = gain
        self.label          = label
    # --- End Init Node --- #
""" END NODE /////////////////////////////////////////////////////////////////////////////////////////////////////// """
