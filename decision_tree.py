import numpy as np

""" Decision Tree ======================================================================================================
    A supervised learning method.
    
    Good References:
    https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf
    https://www.softwaretestinghelp.com/decision-tree-algorithm-examples-data-mining/
"""
class DecisionTree:
    """ Init Tree -------------------------------------------------------------------------------------------------- """
    def __init__(self, min_split=3, depth=3):
        self.dim       = 32*32*3
        self.root      = None
        self.depth     = depth
        self.weights   = np.zeros((self.dim, 1))
        self.min_split = min_split
    # --- End Init Tree --- #


    """ Create Node ------------------------------------------------------------------------------------------------ """
    def create_node(self, label=None, feature_index=None, info_gain=None, threshold=None, leaves=None):
        new_node               = Node()
        new_node.label         = label
        new_node.feature_index = feature_index
        new_node.threshold     = threshold
        new_node.info_gain     = info_gain
        if leaves is not None:
            new_node.add_leaves(leaves)
        return new_node
    # --- End Create Node --- #


    """ Fit -------------------------------------------------------------------------------------------------------- """
    def fit(self, training_data, current_depth):
        # Ref: https://www.youtube.com/watch?v=sgQAhG5Q7iY
        x = training_data[:,:-1]
        y = training_data[:,:-1]
        n_samples, n_features = np.shape(x)

        # split tree in the best way if possible
        if n_samples >= self.min_split and current_depth <= self.depth:
            best_split = self.best_split(training_data, n_samples, n_features)

            # if info gain is positive add onto tree
            if best_split["info_gain"] > 0:
                false_node = self.fit(best_split["data_false"])
                true_node  = self.fit(best_split["data_true"])
                return Node(
                    label=None,
                    feature_index=best_split["feature_index"],
                    info_gain=best_split["info_gain"],
                    threshold=best_split["threshold"],
                    leaves={True: true_node, False: false_node}
                )

        # else return new leaf node
        return Node(label=self.get_leaf_val(y))
    # --- End Create Tree --- #


    """ Best Split ------------------------------------------------------------------------------------------------- """
    def best_split(self, training_data, n_samples, n_features):
        max_gain = -float("inf")
        best_split = {
            "data_false":    None,
            "data_true":     None,
            "feature_index": None,
            "info_gain":     None,
            "threshold":     None
        }

        # for every feature
        for feature_index in range(n_features):
            feature_values = training_data[:n_samples]
            thresholds = np.unique(feature_values)

            for t in thresholds:
                data_left, data_right = self.split(training_data, feature_index, t)
                if len(data_left) > 0 and len(data_right) > 0:

                    y, l, r = training_data[:, -1], data_left[:, -1], data_right[:, -1]
                    gain = self.info_gain(y, l, r)

                    if gain > max_gain:
                        best_split["data_false"]    = data_left
                        best_split["data_true"]     = data_right
                        best_split["feature_index"] = feature_index
                        best_split["info_gain"]     = gain
                        best_split["threshold"]     = t
                        max_gain = gain

        return best_split
    # --- End Best Split --- #


    """ Info Gain -------------------------------------------------------------------------------------------------- """
    def info_gain(self, parent, l, r):
        w_l = len(l) / len(parent)
        w_r = len(r) / len(parent)

        def entropy(x):
            labels = np.unique(x)
            ent    = 0
            for lab in labels:
                pct = len(x[x==lab]) / len(x)
                ent += -pct * np.log2((pct))
            return ent

        return entropy(parent) - (w_l * entropy(l) + w_r * entropy(r))
    # --- End Info Gain --- #


    """ Split ------------------------------------------------------------------------------------------------------ """
    def split(self, data, index, thresh):
        l = np.array([d for d in data if d[index] <= thresh])
        r = np.array([d for d in data if d[index]  > thresh])
        return l, r
    # --- End Split --- #

""" END DECISION TREE ////////////////////////////////////////////////////////////////////////////////////////////// """



""" Node ===============================================================================================================
    A node within the tree structure.
"""
class Node:
    """ Init Node -------------------------------------------------------------------------------------------------- """
    def __init__(self, label=None, feature_index=None, info_gain=None, threshold=None, leaves=None):
        self.label         = label
        self.feature_index = feature_index
        self.threshold     = threshold
        self.info_gain     = info_gain

        #            Label
        #      If no /    \ If yes
        #  False node       True node
        if leaves is not None:
            self.is_leaf = False
            self.leaves = {
                True:  leaves[True],
                False: leaves[False]
            }
        else:
            self.is_leaf = True
            self.leaves = {
                True:  None,
                False: None
            }
    # --- End Init Node --- #


    """ Add Leaves ------------------------------------------------------------------------------------------------- """
    def add_leaves(self, leaves):
        self.is_leaf = False
        self.leaves[True]  = leaves[True]
        self.leaves[False] = leaves[False]
    # --- End Add Leaves --- #

""" END NODE /////////////////////////////////////////////////////////////////////////////////////////////////////// """

import read_dataset as r

model = DecisionTree()
x_train, x_test, y_train, y_test = r.read_dataset("airplane")
data = np.concatenate((x_train, y_train), axis=0)

print("Begin training...")
model.fit(data, 0)
