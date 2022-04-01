import read_dataset as read
import decision_trees as dt

ZIP     = "cifar-10-python.tar.gz"
DATASET = "./dataset/"
AGENTS  = {
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



""" Run Agents ----------------------------------------------------------------------------------------------------- """
def run_agents(dataset, agents_to_run):
    print(dataset)
# --- End Init Agents --- #



""" Main Function -------------------------------------------------------------------------------------------------- """
if __name__ == "__main__":
    to_run = ["DT"]              # agents to test
    init_agents(to_run)          # initialize wanted agents
    run_agents(DATASET, to_run)  # run wanted agents


# --- End Main Function --- #
