import LR_tests as lr

PATH  = "./dataset/"
ZIP   = "cifar-10-python.tar.gz"
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



""" Main Function -------------------------------------------------------------------------------------------------- """
if __name__ == "__main__":

    # Cost
    lr.lr_res(1000, 1, 0.01)

    """    # Accuracy
    for c in CLASSES:
        #rand_rate = random.uniform(0.0001, 0.01)
        lr.lr_acc(c, [10, 50, 100, 1000], [1, 5, 10, 100], 0.01)
    """

# --- End Main Function --- #
