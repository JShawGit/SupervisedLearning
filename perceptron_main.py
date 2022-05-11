from sklearn.metrics import accuracy_score
import read_dataset_tree_percept
import perceptron as pt

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
    pt_model = pt.Perceptron(len(test["data"][0]), len(labels))
    pt_model.fit(train["data"][0], train["labels"][0], 10)

    pred = pt_model.predict([test["data"][0]])
    print(pred)
    print(test["labels"][0])




# --- End Main Function --- #
