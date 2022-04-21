import logistic_regression as lr
import matplotlib.pyplot as plt
import numpy as np
CLASSES = {
    "airplane":   0,
    "automobile": 1,
    "bird":       2,
    "cat":        3,
    "deer":       4,
    "dog":        5,
    "frog":       6,
    "horse":      7,
    "ship":       8,
    "truck":      9
}

""" LR Res --------------------------------------------------------------------------------------------------------- """
def lr_res(iter, st, rate):

    learning_rate = rate
    iterations    = iter
    step          = st

    cost_flag = False
    dim       = 32*32*3

    # ready plot
    plt.clf()
    plt.title("Learning costs with learning rate " + str(round(rate, 6)))
    plt.xlabel("Iterations")
    plt.style.use('classic')
    plt.ylabel("Costs")

    c = 0
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    maxval = -100000.0
    minval = 100000.0

    # get results for each image type
    for type in CLASSES:
        # train the model, get learning results
        lr_model = lr.logistic_regression(iterations, learning_rate, cost_flag, dim, CLASSES[type])
        lr_data  = lr_model.log_reg_model(step)

        # plot costs
        y = np.array(lr_data['costs'])
        x = np.array(range(len(y)))
        plt.plot(x, y, colors[c], label=type)
        c += 1

        val = max(y)
        if val > maxval:
            maxval = val

        val = min(y)
        if val < minval:
            minval = val

        # print plot
        plt.legend(ncol=2, fontsize=10)
        plt.ylim(minval, maxval)
        plt.xlim(0, iter/step)
        plt.savefig("LR_total_costs" + str(learning_rate) + ".png")

# --- End LR Res --- #



""" LR acc --------------------------------------------------------------------------------------------------------- """
def lr_acc(type, iter, st, rate):
    iterations = iter
    step = st

    learning_rate = rate
    cost_flag = True
    dim = 32*32*3

    # train the model, get learning results
    train = []
    test  = []
    for i in range(len(iterations)):
        lr_model = lr.logistic_regression(iterations[i], learning_rate, cost_flag, dim, CLASSES[type])
        lr_data = lr_model.log_reg_model(step[i])
        train.append(lr_data['train_acc'])
        test.append(lr_data['test_acc'])

    # plot train
    y = np.array(train)
    x = np.array(iterations)

    plt.clf()
    plt.plot(x, y, marker="o")
    plt.title("Training accuracy of " + type + " with learning rate " + str(round(rate, 4)))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.savefig("LR_Train_" + type + ".png")

    # plot train
    y = np.array(test)

    plt.clf()
    plt.plot(x, y)
    plt.title("Testing accuracy of " + type + " with learning rate " + str(round(rate, 4)))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.savefig("LR_Test_" + type + ".png")

# --- End LR Res --- #
