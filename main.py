import read_dataset
import read_dataset as read
import decision_trees as dt

ZIP     = "cifar-10-python.tar.gz"
DATASET = "./dataset/"
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
        # [Initializer, Agent object]
    "DT": [dt.DecisionTrees, None],  # decision trees
    "LR": [None, None],  # logistic regression
    "NB": [None, None],  # naive-bayes
    "PT": [None, None]   # perceptron
}

""" Init Agents ---------------------------------------------------------------------------------------------------- """
def init_agents(agents_to_run):
    # for each agent, initialize the item
    for to_run in agents_to_run:
        if AGENTS[to_run][0] is not None:
            AGENTS[to_run][1] = AGENTS[to_run][0]()
# --- End Init Agents --- #



""" Train Agents --------------------------------------------------------------------------------------------------- """
def train_agents(training_data, agents_to_run):
# --- End Train Agents --- #



""" Test Agents ---------------------------------------------------------------------------------------------------- """
def test_agents(test_data, agents_to_run):
    print("TODO")
# --- End Test Agents --- #



""" Main Function -------------------------------------------------------------------------------------------------- """
if __name__ == "__main__":
    # get data to learn with
    (x_train, y_train), (x_test, y_test) = read_dataset.getCifar()
    data_dict = {
        "train": [x_train, y_train],
        "test":  [x_test,  y_test]
    }

    # initialize all wanted agents
    to_run = ["DT"]
    init_agents(to_run)

    # train 1 time
    train_agents(data_dict["train"], to_run)

    # test 1 time
    test_agents(data_dict["test"], to_run)

# --- End Main Function --- #
