import read_dataset_tree_percept as read_dataset
from sklearn.metrics import accuracy_score
import perceptron as pt

""" Main Function -------------------------------------------------------------------------------------------------- """
if __name__ == "__main__":
    """ Good References:
    https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html#download
    """
    # actual data for learning
    test, train = read_dataset.read_dataset()

    """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    # train models
    X = train["data"]
    Y = train["labels"]
    epochs = [10, 100, 500, 1000]
    iterations = 10
    for e in epochs:
        for i in [50, 100, 250, 500, 750, 1000, 5000, 10000, 25000, 50000]:
            print("\n\n--------------------------------------------------------------------")
            print("TESTING FOR e=" + str(e) + " AND i=" + str(i))

            # initialize model
            pt_model = pt.Perceptron(len(test["data"][0]), 10)


            # fit
            pt_model.fit(X[0:i-1], Y[0:i-1], epochs=e)
            pred = pt_model.predict(test["data"])

            # print predictions
            file = open("./perceptron/i_" + str(i) + "_e_" + str(e) + "_predictions.txt", "w")
            file.write(str(pred))
            file.close()

            # print accuracy
            file = open("./perceptron/i_" + str(i) + "_e_" + str(e) + "_predictions_accuracy.txt", "w")
            file.write(str(accuracy_score(test["labels"], pred)))
            file.close()
# --- End Main Function --- #
