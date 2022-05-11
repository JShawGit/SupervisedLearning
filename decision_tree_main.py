import read_dataset_tree_percept as read_dataset
from sklearn.metrics import accuracy_score
import decision_tree as dt

PATH  = "./dataset/"
TRAINING_FILES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
TESTING_FILES = ["test_batch"]
ZIP   = "cifar-10-python.tar.gz"

""" Main Function -------------------------------------------------------------------------------------------------- """
if __name__ == "__main__":
    """ Good References:
    https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html#download
    """

    # actual data for learning
    test, train = read_dataset.read_dataset()

    """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    # train models
    lengths = [10, 50, 100, 500, 1000, 5000, len(train["data"][0])]
    for n in lengths:
        for i in range(5):
            print("\n\n--------------------------------------------------------------------")
            print("TESTING FOR n=" + str(n) + " AND i=" + str(i))

            # initialize model
            dt_model = dt.DecisionTree(max_depth=n-1)

            X = train["data"][0:n-1]
            Y = train["labels"][0:n-1]

            # fit
            dt_model.fit(X, Y)
            pred = dt_model.predict(test["data"])

            # print tree size
            file = open("i_" + str(i) + "_n_" + str(n) + "_treeinfo.txt", "w")
            file.write("Nodes: " + str(dt_model.n_nodes) + "\nLeaves: " + str(dt_model.leaves))
            file.close()

            # print predictions
            file = open("i_" + str(i) + "_n_" + str(n) + "_predictions.txt", "w")
            file.write(str(pred))
            file.close()

            # print accuracy
            file = open("i_" + str(i) + "_n_" + str(n) + "_predictions_accuracy.txt", "w")
            file.write(str(accuracy_score(test["labels"], pred)))
            file.close()

# --- End Main Function --- #
