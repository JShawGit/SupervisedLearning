import read_dataset
import read_dataset as read
import decision_tree as dt
import logistic_regression as lr
import numpy as np
import matplotlib.pyplot as plt

PATH  = "./dataset/"
ZIP   = "cifar-10-python.tar.gz"
FILES = {
    "train": [
        PATH + "data_batch_1",
        PATH + "data_batch_2",
        PATH + "data_batch_3",
        PATH + "data_batch_4",
        PATH + "data_batch_5"
    ],
    "test": [
        PATH + "test_batch"
    ]
}

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
"""""
LABELS:
    0 - airplane
    1 - automobile
    2 - bird
    3 - cat
    4 - deer
    5 - dog
    6 - frog
    7 - horse 
    8 - ship
    9 - truck
"""""
LABELS = {
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




AGENTS  = {
    #     [Initializer, Agent object]
    "DT": [dt.DecisionTree, None],  # decision trees
    "LR": [None, None],  # logistic regression
    "NB": [None, None],  # naive-bayes
    "PT": [None, None]   # perceptron
}

DATASETS = {
    "train": [],
    "test":  []
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
# --- End Train Agents --- #



""" Test Agents ---------------------------------------------------------------------------------------------------- """
def test_agents(test_data, agents_to_run):
    print(test_data)
# --- End Test Agents --- #



""" Main Function -------------------------------------------------------------------------------------------------- """
if __name__ == "__main__":
    # get data to learn with
    (x_train, y_train), (x_test, y_test) = read_dataset.getCifar()
    data_dict = {  # divide by 255 to keep image-colors in uniform range
        "train": [x_train/255.0, y_train.flatten()],
        "test":  [x_test/255.0,  y_test.flatten()]
    }


    # visualize data by plotting images
    fig, ax = plt.subplots(5, 5)
    k = 0

    for i in range(5):
        for j in range(5):
            ax[i][j].imshow(x_train[k], aspect='auto')
            k += 1

    plt.show()


    # initialize all wanted agents
    to_run = ["DT"]
    init_agents(to_run)

    # Logistic Regression
    iterations = 2000
    learning_rate = 0.01 #0.005
    cost_flag = True
    dim = 32*32*3
    lr_model = lr.logistic_regression(iterations, learning_rate, cost_flag, dim, LABELS['airplane'])

    lr_model_data = lr_model.log_reg_model()
    # End Logistic regression

    # train 1 time
    fit_agents(data_dict["train"], to_run, 100, 100)

    # test 1 time
    test_agents(data_dict["test"], to_run)

# --- End Main Function --- #
