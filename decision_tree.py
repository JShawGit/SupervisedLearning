import numpy as np

""" Decision Tree ======================================================================================================
    A supervised learning method.
"""
class DecisionTree:
    """ Init Tree -------------------------------------------------------------------------------------------------- """
    def __init__(self, min_split=0, depth=0):
        self.root = None
        self.depth = depth
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


    """ Create Tree ------------------------------------------------------------------------------------------------ """
    def create_tree(self, training_data, current_depth):
        # Ref: https://www.youtube.com/watch?v=sgQAhG5Q7iY
        x = training_data[0]
        y = training_data[1]
        n_samples, n_features = np.shape(x)

        # split tree in the best way if possible
        if n_samples >= self.min_split and current_depth <= self.depth:
            best_split = self.best_split(training_data, n_samples, n_features)

            # if info gain is positive add onto tree
            if best_split["info_gain"] > 0:
                false_node = self.create_tree(best_split["data_false"])
                true_node  = self.create_tree(best_split["data_true"])
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
        max_info_gain = -float("inf")
        best_split = {
            "data_false":    None,
            "data_true":     None,
            "feature_index": None,
            "info_gain":     None,
            "threshold":     None
        }

        # for every feature
        for feature_index in range(n_features):
            feature_values = [] # TODO
            thresholds = np.unique(feature_values)

            for t in thresholds:
                data_left, data_right = self.split(training_data, feature_index, t)
                if len(data_left) > 0 and len(data_right) > 0:
                    return None # TODO

        return best_split
    # --- End Best Split --- #


    """ Fit -------------------------------------------------------------------------------------------------------- """
    def fit(self, training_data, epochs, batch_size):
        assert training_data[0].shape[0] == len(training_data[1]), \
            "Training data array length not equal to validation array!"
        print("TODO")
    # --- End Fit --- #

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
                True:  None,\
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
