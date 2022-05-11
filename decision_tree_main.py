from sklearn.metrics import accuracy_score
import read_dataset_tree_percept
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
    train, test = read_dataset.read_dataset(PATH)
    train = read_dataset.preprocessing(train)
    test  = read_dataset.preprocessing(test)
    test = {
        "data":   train["data"].pop(),
        "labels": train["labels"].pop()
    }

    # get data labels
    batch_meta = read_dataset.unpickle(PATH + "batches.meta")
    labels = batch_meta['label_names']

    """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    # train models
    lengths = [10, 50, 100, 500, 1000]
    depths = [9, 49, 99, 499, 999]
    for n in lengths:
        for d in depths:
            if d >= n:
                continue

            print("\n\n--------------------------------------------------------------------")
            print("TESTING FOR n=" + str(n) + " AND depth=" + str(d))

            # initialize model
            dt_model = dt.DecisionTree(max_depth=d)

            # fit
            dt_model.fit(train["data"][0][0:n], train["labels"][0][0:n])
            pred = dt_model.predict(test["data"])

            # print tree size
            file = open("n_" + str(n) + "_d_" + str(d) + "_treeinfo.txt", "w")
            file.write("Nodes: " + str(dt_model.n_nodes) + "\nLeaves: " + str(dt_model.leaves))
            file.close()

            # print predictions
            file = open("n_" + str(n) + "_d_" + str(d) + "_predictions.txt", "w")
            file.write(str(pred))
            file.close()

            # print accuracy
            file = open("n_" + str(n) + "_d_" + str(d) + "_predictions_accuracy.txt", "w")
            file.write(str(accuracy_score(test["labels"], pred)))
            file.close()




"""

        
        
    # train models
    lengths = [10, 50, 100, 500, 1000, 5000, len(train["data"][0])]
    for n in lengths:
        for i in range(len(train["data"])):
            print("\n\n--------------------------------------------------------------------")
            print("TESTING FOR n=" + str(n) + " AND i=" + str(i))

            # initialize model
            dt_model = dt.DecisionTree(max_depth=n-1)

            X = train["data"][i][0:n]
            Y = train["labels"][i][0:n]

            X_new = []

            for x in range(len(X)):
                X_new.append(sum(X[x]) / len(X[x]))

            print(X_new)

            # fit
            dt_model.fit(X_new, Y)
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
    """




# --- End Main Function --- #
