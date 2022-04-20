import read_dataset
import read_dataset as read
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
    "DT": [None, None],  # decision trees
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
# --- End Fit Agents --- #



""" LR Res --------------------------------------------------------------------------------------------------------- """
def lr_res(type, iter, st, rate):
    iterations = iter
    step = st

    learning_rate = rate
    cost_flag = True
    dim = 32*32*3

    # train the model, get learning results
    lr_model = lr.logistic_regression(iterations, learning_rate, cost_flag, dim, LABELS[type])
    lr_data = lr_model.log_reg_model(step)

    # plot costs
    y = np.array(lr_data['costs'])
    x = np.array(range(len(y)))
    plt.plot(x, y)
    plt.title("Learning costs of " + type + " with learning rate " + str(rate))
    plt.xlabel("Iterations")
    plt.ylabel("Costs")
    plt.savefig("LR_" + type + ".png")
# --- End LR Res --- #



""" Main Function -------------------------------------------------------------------------------------------------- """
if __name__ == "__main__":

    # make graphs
    for c in CLASSES:
        lr_res(c, 1000, 50, 0.1)

# --- End Main Function --- #
