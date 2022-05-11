import random

import matplotlib.pyplot as plt
import pandas as pd
import LR_tests as lr
import naive_bayes as NB

PATH = "./dataset/"
ZIP = "cifar-10-python.tar.gz"
AGENTS = {
    #     [Initializer, Agent object]
    "DT": [None, None],  # decision trees
    "LR": [None, None],  # logistic regression
    "NB": [None, None],  # naive-bayes
    "PT": [None, None]  # perceptron
}
DATASETS = {
    "train": [],
    "test": []
}

""" Init Agents ---------------------------------------------------------------------------------------------------- """


def init_agents(agents_to_run):
    for run in agents_to_run:
        if AGENTS[run][0] is not None:
            AGENTS[run][1] = AGENTS[run][0]()


# --- End Init Agents --- #


""" Fit Agents ----------------------------------------------------------------------------------------------------- """


def fit_agents(training_data, agents_to_run, epochs, batch_size):
    for run in agents_to_run:
        if AGENTS[run][0] is not None:
            AGENTS[run][1].fit(training_data, epochs, batch_size)


# --- End Fit Agents --- #

def plot_bayes(test_acc, train_acc, py_method_test, py_method_train, class_num):
    x = False
    x_axis = [1,2,3,4,5,6,7,8,9]

    if x:
        # ready plot
        plt.figure()
        plt.plot(x_axis, test_acc, label='Test Data Set Accuracy')
        plt.plot(x_axis, train_acc, label='Train Data Set Accuracy')
        plt.plot(x_axis, py_method_test, label='Scikit-learn: Test Data Set Accuracy')
        plt.plot(x_axis, py_method_train, label='Scikit-learn: Train Data Set Accuracy')
        plt.title("Naive Bayes: Accuracy of Predictions as a Function of the Number of Classification Types")
        plt.xlabel("Number of Image Classification Types")
        plt.style.use('classic')
        plt.ylabel("Accuracy [%]")
        plt.legend()
        plt.show()
        plt.savefig("NB_Acc_Func_Class.png")

    else:
        plotdata = pd.DataFrame({

             "SkLearn": py_method_test,

            "Our Method": test_acc},

            index=["airplane", "automobile", "bird", "cat", 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

        plotdata.plot(kind="bar", figsize=(15, 8))

        plt.title("NaiveBayes: Accuracy of Predictions as a Function of Classification Type")

        plt.xlabel("Image Types")

        plt.ylabel("Accuracy Rate [%]")

        min_val_0 = min(py_method_test)
        min_val_1 = min(test_acc)

        plt.ylim(min(min_val_0, min_val_1) - 4, 100)
        plt.show()
        print()


""" Main Function -------------------------------------------------------------------------------------------------- """
if __name__ == "__main__":
    # Cost
    # lr.lr_res(500, 10, 0.1)
    # lr.lr_res(500, 10, 0.01)
    # lr.lr_res(500, 10, 0.005)
    # lr.lr_res(500, 10, 0.001)

    # lr.lr_acc(500, 10, 0.1)
    # lr.lr_acc(500, 10, 0.01)
    # lr.lr_acc(500, 10, 0.001)
    # lr.lr_acc(500, 10, 0.0001)

    """    # Accuracy
    for c in CLASSES:
        #rand_rate = random.uniform(0.0001, 0.01)
        lr.lr_acc(c, [10, 50, 100, 1000], [1, 5, 10, 100], 0.01)
    """

    CLASSES = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9
    }
    # [1, 2, 4, 8, 16, 32]
    dim = 1
    # Batch Number [1,2,3,4,5]
    binary_class = True
    #num_class = [1+1,2+1,3+1,4+1,5+1,6+1,7+1,8+1,9+1] #[1+1] # the number of clasifications you want to make
    num_class = [1+1]
    accuracy = []


    for class_sel in CLASSES:
        naive_bayes = NB.naive_bayes(dim, CLASSES[class_sel], binary_class, 2)
        accuracy.append(naive_bayes.naive_bayes_run())
    print()
    '''''
    for num in num_class:
        naive_bayes = NB.naive_bayes(dim, CLASSES['truck'], binary_class, num)
        accuracy.append(naive_bayes.naive_bayes_run())
    print()
    '''''


    test_acc = []
    train_acc = []
    test_acc_skl = []
    train_acc_skl = []
    num_class = [2,3,4,5,6,7,8,9,10,11]

    for num in num_class:
        test_acc.append(accuracy[num - 2][0])
        train_acc.append(accuracy[num - 2][1])
        test_acc_skl.append(accuracy[num - 2][2])
        train_acc_skl.append(accuracy[num - 2][3])

    plot_bayes(test_acc, train_acc, test_acc_skl, train_acc_skl,num_class)
# --- End Main Function --- #
