# read the data ---Alice

from Structures import Node, Node.split


def compute_gini_index([good, bad]):
    """
    This function calculates the gini index on a new split,
    it takes the labels that are gone in the considered split and
    returns the gini index (or impurity).
    This function is used as a subroutine in the function compute_best_split.

    :param array: array of classifications.
    :return: the gini index of the split.
    """
    prob_good = count(good)/(count(good)+count(bad))
    prob_bad = count(bad)/(count(good)+count(bad))

    gini = (prob_good*prob_bad)-((count(Node.left.bad)/count(good)+count(bad))+(count(Node.right.bad)/count(good)+count(bad))

    return gini

# compute the best split

def compute_best_split(x_data: [[]], y_data: [], minleaf: int):
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

    best_split = min(compute_gini_index())

    return best_split