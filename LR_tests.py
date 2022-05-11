import logistic_regression as lr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

""" LR Res --------------------------------------------------------------------------------------------------------- """


def lr_res(iter, st, rate):
    learning_rate = rate
    iterations = iter
    step = st

    cost_flag = False
    dim = 32 * 32 * 3

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
        lr_data = lr_model.log_reg_model(step)

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
        plt.xlim(0, iter / step)
        #plt.xtickformat('%g0')
        plt.savefig("LR_total_costs" + str(learning_rate) + ".png")


# --- End LR Res --- #


""" LR acc --------------------------------------------------------------------------------------------------------- """


def lr_acc(iter, st, rate):
    learning_rate = rate
    iterations = iter
    step = st

    cost_flag = False
    dim = 32 * 32 * 3

    # ready plot
    # plt.clf()
    # plt.title("Success Rates for learning rate " + str(round(rate, 6)))
    # plt.xlabel("Image Types")
    # plt.style.use('classic')
    # plt.ylabel("Success Rate [%]")

    c = 0
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    maxval = -100000.0
    minval = 100000.0

    train_acc = []
    test_acc = []
    # get results for each image type
    for type in CLASSES:
        # train the model, get learning results
        lr_model = lr.logistic_regression(iterations, learning_rate, cost_flag, dim, CLASSES[type])
        lr_data = lr_model.log_reg_model(step)

        # plot costs
        train_acc.append(lr_data['train_acc'])
        test_acc.append(lr_data['test_acc'])

    plotdata = pd.DataFrame({

        "Test": train_acc,

        "Train": test_acc},

        index=["airplane", "automobile", "bird", "cat", 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    plotdata.plot(kind="bar", figsize=(15, 8))

    plt.title("Success Rates for learning rate " + str(rate))

    plt.xlabel("Image Types")

    plt.ylabel("Success Rate [%]")

    min_val_0 = min(train_acc)
    min_val_1 = min(test_acc)

    plt.ylim(min(min_val_0, min_val_1)-4, 100)

    # print plot
    plt.savefig("LR_total_acc" + str(learning_rate) + ".png")

# --- End LR Res --- #
