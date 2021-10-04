import pandas as pd
import numpy as np
from sklearn import metrics


def read_data(path_data_train, path_data_test):
    """
    This function is used to open the data using a pandas Dataframe.

    :param path_data_train: the complete path in which the file containing the training data is located on a PC.
    :param path_data_test: the complete path in which the file containing the testing data is located on a PC.
    :return: both the data_train and data_test opened as a pandas Dataframe.
    """

    if path_data_train.endswith("credit.txt") or path_data_train.endswith("pima.txt"):
        data_train = pd.read_csv(path_data_train, sep=',')
        data_test = pd.read_csv(path_data_test, sep=',')


    else:
        data_temp_train = pd.read_csv(path_data_train, sep=';')
        data_temp_test = pd.read_csv(path_data_test, sep=';')
        data_train = data_temp_train[["pre", "post", "ACD_avg", "ACD_max", "ACD_sum", "FOUT_avg", "FOUT_max", "FOUT_sum", "MLOC_avg", "MLOC_max", "MLOC_sum",
                                "NBD_avg", "NBD_max", "NBD_sum", "NOCU", "NOF_avg", "NOF_max", "NOF_sum", "NOI_avg", "NOI_max", "NOI_sum",
                                "NOM_avg", "NOM_max", "NOM_sum", "NOT_avg", "NOT_max", "NOT_sum", "NSF_avg", "NSF_max", "NSF_sum", "NSM_avg",
                                "NSM_max", "NSM_sum", "PAR_avg", "PAR_max", "PAR_sum", "TLOC_avg", "TLOC_max", "TLOC_sum", "VG_avg", "VG_max",
                                "VG_sum"]]
        data_test = data_temp_test[["pre", "post", "ACD_avg", "ACD_max", "ACD_sum", "FOUT_avg", "FOUT_max", "FOUT_sum", "MLOC_avg", "MLOC_max", "MLOC_sum",
                                "NBD_avg", "NBD_max", "NBD_sum", "NOCU", "NOF_avg", "NOF_max", "NOF_sum", "NOI_avg", "NOI_max", "NOI_sum",
                                "NOM_avg", "NOM_max", "NOM_sum", "NOT_avg", "NOT_max", "NOT_sum", "NSF_avg", "NSF_max", "NSF_sum", "NSM_avg",
                                "NSM_max", "NSM_sum", "PAR_avg", "PAR_max", "PAR_sum", "TLOC_avg", "TLOC_max", "TLOC_sum", "VG_avg", "VG_max",
                                "VG_sum"]]

    return data_train, data_test


def split_label(data, file):
    """
    This function will split the label column from the attributes columns of given data.

    :param data: data that have to be split.
    :param file: the filepath in which the data are located on a PC.
    :return: data_x which is an array of attributes and data_y which are the corresponding labels.
    """

    data_copy = data.copy(deep=True)

    if file.endswith("credit.txt"):
        data_y = np.array(data_copy["class"])
        data_copy.drop("class", inplace=True, axis=1)
        data_x = np.array(data_copy)

    elif file.endswith("pima.txt"):
        data_y = np.array(data_copy["class"])
        data_copy.drop("class", inplace=True, axis=1)
        data_x = np.array(data_copy)

    else:
        data_y = np.array(data_copy["post"])
        data_y[data_y > 0] = 1
        data_copy.drop("post", inplace=True, axis=1)
        data_x = np.array(data_copy)

    del data_copy

    return data_x, data_y


def get_attributes_list(x_train, data_train):
    """
    This function encodes the attribute given in input in integer values, and makes
    a dictionary to map this encoding.

    :param x_train: array of attributes.
    :param data_train: pandas dataframe of complete data.
    :return: a dictionary that specifies the attributes of the given data.
    Format: attributes = {0: "attribute_0", 1: "attribute_1", ...}.
    """

    attributes = {}
    data_copy = data_train.copy(deep=True)

    if "post" in data_copy:
        data_copy.drop("post", inplace=True, axis=1)

    attributes_list = range(len(x_train[0]))

    for i in attributes_list:
        attributes[i] = data_copy.columns[i]

    del data_copy

    return attributes


def compute_gini_index(labels: []):
    """
    This function calculates the gini index on a new split,
    it takes the labels that are gone in the considered split and
    returns the gini index (or impurity).
    This function is used as a subroutine in the function compute_best_split.

    :param labels: array of classifications.
    :return: the gini index of the split.
    """

    total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    count_0_and_1 = dict(zip(unique, counts))

    # if nothing was classified as 0:
    if 0 not in count_0_and_1:
        count_0_and_1[0] = 0

    p_0 = count_0_and_1[0]/total

    gini_index = p_0 * (1 - p_0)

    return gini_index


def best_split_one_attribute(data_x: [], data_y: [], minleaf: int):
    """
    This function will compute, within the values of a given attribute, which is the best
    one to perform a split.

    :param data_x: all the values of the attribute given in input.
    :param data_y: the labels of the attribute given in input.
    :param minleaf: minimum number of observations (elements) in order to create a leaf.
    :return: the value of the best split for this attribute, impurities for both children generated, and
    percentage of cases that are gone to the left child and to the right child.
    """

    lowest_impurity = 1
    best_impurity_left, best_impurity_right = 1, 1
    best_split = None
    best_percentage_left, best_percentage_right = 1, 0  # initialized as all cases will go to the left child

    sorted_values = np.sort(np.unique(data_x))
    len_sorted = len(sorted_values)
    # generate all the values 'in between' the given values:
    possible_splits = (sorted_values[0:(len_sorted - 1)] + sorted_values[1:len_sorted]) / 2
    # Eg: if sorted = [1, 2, 4] --> possible_splits = [1.5, 3]
    sorted_indices = data_x.argsort()

    data_x = data_x[sorted_indices]
    data_y = data_y[sorted_indices]

    for i in range(len(possible_splits)):

        left_data = data_y[data_x <= possible_splits[i]]
        right_data = data_y[data_x > possible_splits[i]]

        percentage_left = len(left_data)/len(data_y)
        percentage_right = len(right_data)/len(data_y)

        impurity_left = compute_gini_index(left_data)
        impurity_right = compute_gini_index(right_data)

        children_impurity = impurity_left * percentage_left + impurity_right * percentage_right

        if children_impurity < lowest_impurity and len(left_data) >= minleaf and len(right_data) >= minleaf:
            # store the new best values found by far:
            best_impurity_left = impurity_left
            best_impurity_right = impurity_right
            lowest_impurity = children_impurity
            best_percentage_left = percentage_left
            best_percentage_right = percentage_right
            best_split = possible_splits[i]

    return best_split, best_impurity_left, best_impurity_right, best_percentage_left, best_percentage_right


def best_split_all_attributes(data_x: [[]], data_y: [], minleaf: int, attributes: dict):
    """
    This function uses the gini-index to calculate the impurity of a split, with the objective
    to understand if a split should be made on a Node.
    Basically, for each attribute, it finds the best split and, among these, will
    select and return the best one.

    :param data_x: array of values of an attribute to perform a split.
    :param data_y: array of classifications in the same order of data_x.
    :param minleaf: minimum number of observations (elements) in order to create a leaf.
    Basically, it specifies the minimum number of observations (elements) that have to be
    in each part of the split in order for a it to be considered valid.
    :return: best attribute on which we have split, value of the best split
    and impurities for both parts of the split.
    """

    lowest_impurity = 1
    best_impurity_left, best_impurity_right = 0, 0
    best_split = None
    best_attribute = None

    # a split can be made only on attributes that have not yet been used!
    for i in attributes:

        new_split, impurity_left, impurity_right, percentage_left, percentage_right = best_split_one_attribute(data_x[:, i], data_y, minleaf)
        children_impurity = impurity_left * percentage_left + impurity_right * percentage_right

        if children_impurity < lowest_impurity:
            best_impurity_left = impurity_left
            best_impurity_right = impurity_right
            lowest_impurity = children_impurity
            best_split = new_split
            best_attribute = i

    return best_attribute, best_split, best_impurity_left, best_impurity_right


def print_tree(node, attributes, level: int = 0):
    """
    This function will print the whole tree.

    :param node: node to start the search from the tree, Should be root
    :param level: auxiliar variable in order to know the depth level due to the recursive nature of the function
    :return: level of depth of the moment
    """
    print("Level:", level)
    if node.is_leaf:
        print('Node is a leaf, classification:', str(node.classification))
    else:
        print("Intermediate node, attribute and value:", attributes[node.attribute], "<=", node.value)

    level += 1

    if node.left is not None:
        print_tree(node.left, attributes, level)

    if node.right is not None:
        print_tree(node.right, attributes, level)

    return level


def evaluation(data_y: [], predictions: []):
    """
    This function will calculate the accuracy, precision and recall by comparing
    the predictions obtained by a certain model with the actual values.

    :param data_y: array of classifications.
    :param predictions: array of predicted values.
    :return: accuracy, precision and recall.
    """

    print("\nReport:")
    print(metrics.classification_report(data_y, predictions))

    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix(data_y, predictions))

    accuracy = metrics.accuracy_score(data_y, predictions)
    print("\nAccuracy:", accuracy)

    precision = metrics.precision_score(data_y, predictions)
    print("Precision:", precision)

    recall = metrics.recall_score(data_y, predictions)
    print("Recall:", recall)

    return accuracy, precision, recall
