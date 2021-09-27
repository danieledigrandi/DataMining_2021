# read the data ---Alice

import pandas as pd

def read_data(file="credit"):
    """
    This function is used to open the data using a pandas Dataframe.
    :param file: the file to open. Possible values: "credit", "pima", "eclipse-train", "eclipse-test".
    :return: the opened data as a pandas Dataframe.
    """

    if file == "credit":
        file = "./Assignment_1/data/credit.txt"
        data = pd.read_csv(file)

    elif file == "pima":
        file = "./Assignment_1/data/pima.txt"
        data = pd.read_csv(file)

    elif file == "eclipse-train":
        file = "./Assignment_1/data/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv"
        data = pd.read_csv(file, sep=';')

    elif file == "eclipse-test":
        file = "./Assignment_1/data/promise-2_0a-packages-csv/eclipse-metrics-packages-3.0.csv"
        data = pd.read_csv(file, sep=';')

    return data



# def compute_gini_index():
    """
    This function calculates the gini index on a new split,
    it takes the labels that are gone in the considered split and
    returns the gini index (or impurity).
    This function is used as a subroutine in the function compute_best_split.

    :param array: array of classifications.
    :return: the gini index of the split.
    """
    # prob_good = count(good)/(count(good)+count(bad))
    # prob_bad = count(bad)/(count(good)+count(bad))

    # gini = (prob_good*prob_bad)-((count(Node.left.bad)/count(good)+count(bad))+(count(Node.right.bad)/count(good)+count(bad))

    # return gini


# compute the best split

# def compute_best_split(x_data: [[]], y_data: [], minleaf: int):
    """
    This function uses the gini-index to calculate the impurity reduction of a split, with the objective
    to understand if a split should be made on a Node.
    Basically, it finds the possible splits and returns the split with the greater impurity reduction,
    together with other values.

    :param x_data: array of values of an attribute to perform a split.
    :param y_data: array of classifications in the same order of data_x.
    :param minleaf: minimum number of observations (elements) in order to create a leaf.
    Basically, it specifies the minimum number of observations (elements) that have to be
    in each part of the split in order for a it to be considered valid.
    :return: value of the split, impurities for both parts of the split, and fraction of observations
    that are gone in both splits and attributes on which we have split.
    """

    """
    At a certain point there will be something like:
    for i in attributes:
        new_split, impurity_1, impurity_2, len_1, len_2 = compute_best_split(data_x[:, i], data_y, minleaf)
    """

    # in this function, use the compute_gini_index function to compute the
    # gini index (or impurity) of a split.

    # best_split = min(compute_gini_index())

    # return best_split