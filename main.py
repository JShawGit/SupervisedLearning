import read_dataset
import read_dataset as read
import decision_tree as dt

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
        "train": [x_train/255.0, y_train/255.0],
        "test":  [x_test/255.0,  y_test/255.0]
    }

    # initialize all wanted agents
    to_run = ["DT"]
    init_agents(to_run)

    # train 1 time
    fit_agents(data_dict["train"], to_run, 100, 100)

    # test 1 time
    test_agents(data_dict["test"], to_run)

# --- End Main Function --- #
