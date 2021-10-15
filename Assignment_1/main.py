# Data mining project 2021 - assignment 1

from math import sqrt
import numpy as np
import random
import pandas as pd
from sklearn import metrics
from time import time
from statsmodels.stats.contingency_tables import mcnemar

"""
---------------------------------------------------------------------------------------------

Project developed by:
- Di Grandi Daniele
- Hartkamp Jens
- Hartog Alice

---------------------------------------------------------------------------------------------

READ ME:
When starting the program, remember to insert the complete path
of the data that have to be opened.
The program supports the credit.txt, pima.txt and eclipse (all the .csv) data.
The 4 functions (tree_grow, tree_pred, tree_grow_b and tree_pred_b) are not at the beginning
because, as we implemented them, they need the classes to work. 
Hence, the code is structured in this way:

- Classes definition (Node, Tree and Forest)
- The 4 functions (tree_grow, tree_pred, tree_grow_b and tree_pred_b)
- Rest of the functions

Every function is commented.

---------------------------------------------------------------------------------------------

From GitHub:
FILEPATH_MAC:
credit: ./Assignment_1/data/credit.txt
pima: ./Assignment_1/data/pima.txt
eclipse-train: ./Assignment_1/data/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv
eclipse-test: ./Assignment_1/data/promise-2_0a-packages-csv/eclipse-metrics-packages-3.0.csv

FILEPATH_WINDOWS:
credit: ./data/credit.txt
pima: ./data/pima.txt
eclipse-train: ./data/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv
eclipse-test: ./data/promise-2_0a-packages-csv/eclipse-metrics-packages-3.0.csv

---------------------------------------------------------------------------------------------

"""


class Node:
    """
    Class created to define a Node object, it belongs to a Tree and it can be: root, leaf or an intermediate Node.
    When the Tree is growing, new nodes are generated and recursively classified.
    """

    def __init__(self, root: bool = False, leaf: bool = False, classification: int = None):
        """
        Constructor for the class Node: root Node means that the Node is the beginning of the tree.
        A Node can have a right Node and a left Node (children), if it is not a leaf.
        If the Node is a leaf it has the value of the classification (1, 0): if the value is None,
        then it is not a leaf and it can be split, hence, it has the attribute used to divide the data,
        which is an int encoded as: 0: attribute_0, 1: attribute_1, ...
        and the value of the split (e.g. split the data based on a numerical attribute: < 36.5, then, 36.5 is the
        value of the split).

        :param root: True if node is root, False otherwise.
        :param leaf: True if node is leaf (pure), False otherwise.
        :param classification: if leaf == True, here there is the value of classification (binary).
        """

        self.is_root = root
        self.is_leaf = leaf
        self.classification = classification
        self.value = None
        self.attribute = None
        self.left = None
        self.right = None


    def split(self, data_x: [[]], data_y: [], nmin: int, minleaf: int, nfeat: int, attributes: dict, rf_mode: int):
        """
        This function made the Tree grows by splitting the data in input into 2 different groups.
        To do this, it iterates through nfeat attributes and calculate the best split of each.
        Then, the attribute that gives an impurity reduction greater than the others,
        would be selected as attribute to split the data, and it is saved together with its split value.
        Then, if one of the child nodes classifies as a leaf, it generates a leaf Node.
        Otherwise, it generates a normal Node.

        :param data_x: array of data attributes and values passed without the classification labels.
        :param data_y: array of classifications in the same order of data_x.
        :param nmin: minimum number of observations (elements) in order to create a Node.
        :param minleaf: minimum number of observations (elements) in order to create a leaf.
        :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
        :param attributes: a dictionary that specifies the attributes available to make a split.
        Format: attributes = {0: "attribute_0", 1: "attribute_1", ...}.
        :param rf_mode: whether the random forest mode is active. 0 means no, 1 means that the computation will be performed
        using the random forest method. This is because in this case, we can select nfeat attributes
        each time we are going to make a split.
        :return: None, it simply stores in its variables the created nodes and their connections.
        """

        if rf_mode:
            random_forest_attributes = random.sample(list(attributes), nfeat)
            best_attribute, best_split, best_impurity_left, best_impurity_right = best_split_all_attributes(data_x, data_y, minleaf, random_forest_attributes)

            # if maybe we had bad luck with the first attributes selection, try again with nfeat different attributes
            # until a certain threshold value of 300 iterations was reached:
            threshold = 300
            num_iter = 0
            while best_attribute is None and best_split is None and num_iter < threshold:
                random_forest_attributes = random.sample(list(attributes), nfeat)
                best_attribute, best_split, best_impurity_left, best_impurity_right = best_split_all_attributes(data_x, data_y, minleaf, random_forest_attributes)
                num_iter += 1
        else:
            best_attribute, best_split, best_impurity_left, best_impurity_right = best_split_all_attributes(data_x, data_y, minleaf, attributes)


        self.attribute = best_attribute
        self.value = best_split


        # if no split is allowed on the given attributes, or the node contains less than nmin cases, then the node is a leaf.
        # the only node that can contain less then nmin cases at this point, is the root node, since in further checks,
        # a child node is immediately transformed in a leaf if the cases that will contain are less than nmin.
        if self.attribute is None and self.value is None or len(data_x) < nmin:
            unique, counts = np.unique(data_y, return_counts=True)
            count_0_and_1 = dict(zip(unique, counts))

            # if nothing was classified as 0:
            if 0 not in count_0_and_1:
                count_0_and_1[0] = 0

            # if nothing was classified as 1:
            if 1 not in count_0_and_1:
                count_0_and_1[1] = 0

            # if the number of 1s is equal to number of 0s: classify as 0
            # else, classify with the major class.
            if count_0_and_1[0] == count_0_and_1[1]:
                classification = 0
            else:
                classification = max(count_0_and_1, key=count_0_and_1.get)

            self.is_leaf = True
            self.classification = classification

        else:
            # if we have a split and attribute value,
            # sort the data left and right based on the best split
            left_node_data = data_x[:, best_attribute] <= best_split
            right_node_data = data_x[:, best_attribute] > best_split

            # checking the left child

            if best_impurity_left == 0 or len(data_x[left_node_data]) < nmin:

                unique, counts = np.unique(data_y[left_node_data], return_counts=True)
                count_0_and_1 = dict(zip(unique, counts))

                # if nothing was classified as 0:
                if 0 not in count_0_and_1:
                    count_0_and_1[0] = 0

                # if nothing was classified as 1:
                if 1 not in count_0_and_1:
                    count_0_and_1[1] = 0

                # if the number of 1s is equal to number of 0s: classify as 0
                # else, classify with the major class.
                if count_0_and_1[0] == count_0_and_1[1]:
                    classification = 0
                else:
                    classification = max(count_0_and_1, key=count_0_and_1.get)

                self.left = Node(leaf=True, classification=classification)

            else:
                self.left = Node()
                self.left.split(data_x[left_node_data], data_y[left_node_data], nmin, minleaf, nfeat, attributes, rf_mode)

            # checking the right child

            if best_impurity_right == 0 or (len(data_x[right_node_data]) < nmin):

                unique, counts = np.unique(data_y[right_node_data], return_counts=True)
                count_0_and_1 = dict(zip(unique, counts))

                # if nothing was classified as 0:
                if 0 not in count_0_and_1:
                    count_0_and_1[0] = 0

                # if nothing was classified as 1:
                if 1 not in count_0_and_1:
                    count_0_and_1[1] = 0

                # if the number of 1s is equal to number of 0s: classify as 0
                # else, classify with the major class.
                if count_0_and_1[0] == count_0_and_1[1]:
                    classification = 0
                else:
                    classification = max(count_0_and_1, key=count_0_and_1.get)

                self.right = Node(leaf=True, classification=classification)

            else:
                self.right = Node()
                self.right.split(data_x[right_node_data], data_y[right_node_data], nmin, minleaf, nfeat, attributes, rf_mode)


    def classify(self, data: []):
        """
        Recursive method that finds the correct leaf for new data, it sends the data
        to the predicted child node returning the classification value for those data.

        :param data: array of data attributes and values to be classified, passed without the classification labels.
        :return: classification labels of the data array received in input.
        """

        if self.is_leaf:
            return self.classification
        else:
            if data[self.attribute] < self.value:
                return self.left.classify(data)
            else:
                return self.right.classify(data)


class Tree:
    """
    Class created to define a Tree object, and it simply has a pointer to a Node,
    which is the root of the Tree.
    From the root, as in the classic binary Tree data-structure, all the stored data are accessible
    through pointers to the left or right child nodes.
    """

    def __init__(self):
        """
        Constructor for the class Tree that generates a root Node object.
        """

        self.root = Node(root=True)


    def grow(self, data_x: [[]], data_y: [], nmin: int, minleaf: int, nfeat: int):
        """
        This function grows the tree using the x_train data (here: data_x) together with their labels y_train (here: data_y).
        How the tree grows: recursively, it calls the method split from the class Node.

        :param data_x: array of data attributes and values passed without the classification labels.
        :param data_y: array of classifications in the same order of x.
        :param nmin: minimum number of observations (elements) in order to create a Node.
        :param minleaf: minimum number of observations (elements) in order to create a leaf.
        :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
        :return: Tree created using the training set.
        """

        self.root.split(data_x, data_y, nmin=nmin, minleaf=minleaf, nfeat=nfeat, attributes=attributes, rf_mode=rf_mode)


    def classify(self, data_x: [[]]):
        """
        This function uses a single Tree to classify the data received in input.

        :param data_x: data to be classified.
        :return: classification of the data.
        """

        predictions = []

        for observation in data_x:
            predictions.append(self.root.classify(observation))

        return predictions


class Forest:
    """
    Class created to define a Forest object. A Forest is defined as a List of Trees.
    """

    def __init__(self):
        """
        Constructor for the class Forest.
        """
        self.forest = []
        self.classification = []


    def grow_b(self, x: [[]], y: [], nmin: int, minleaf: int, nfeat: int, m: int):
        """
        This function let grows a Forest.
        It can be used also to let grow a Bagging.

        :param x: array of data attributes and values passed without the classification labels.
        :param y: array of classifications in the same order of x.
        :param nmin: minimum number of observations (elements) in order to create a Node.
        :param minleaf: minimum number of observations (elements) in order to create a leaf.
        :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
        :param m: number of trees to grow in the current forest.
        :return: Forest created using the training set.
        """

        # averaging with bootstrapping: randomly select the used data
        for i in range(m):
            new_x_data, new_y_data = np.empty_like(x), np.empty_like(y)
            for j in range(len(y)):
                observation = random.randint(0, len(y)-1)
                new_x_data[j] = x[observation]
                new_y_data[j] = y[observation]

            tree = Tree()
            tree.grow(new_x_data, new_y_data, nmin, minleaf, nfeat)
            self.forest.append(tree)

            if i % 5 == 0 and i != 0:
                print("Trees evaluated by far:", i, "/", m)
            if i == m - 1:
                print("Trees evaluated by far:", i + 1, "/", m)

        return self.forest


    def pred_b(self, data, trees):
        """
        This function uses a Forest (or Bagging) to classify the data received in input.

        :param data: data to be classified.
        :param trees: list of Trees used to classify the data.
        :return: classification of the data.
        """

        self.classification = np.zeros(len(data))
        count = []

        for tree in trees:
            count.append(tree.classify(data))
        count = np.array(count)

        for column in range(len(count[0])):

            unique, counts = np.unique(count[:, column], return_counts=True)
            count_0_and_1 = dict(zip(unique, counts))

            # if nothing was classified as 0:
            if 0 not in count_0_and_1:
                count_0_and_1[0] = 0

            # if nothing was classified as 1:
            if 1 not in count_0_and_1:
                count_0_and_1[1] = 0

            # if the number of 1s is equal to number of 0s: classify as 0
            # else, classify with the major class.
            if count_0_and_1[0] == count_0_and_1[1]:
                classification = 0
            else:
                classification = max(count_0_and_1, key=count_0_and_1.get)

            self.classification[column] = classification

        return self.classification


# ---------------------------------------------------------------------------------------------


def tree_grow(x: [[]], y: [], nmin: int, minleaf: int, nfeat: int):
    """
    This function creates a Tree object tailored for the given data.

    :param x: array of data attributes and values passed without the classification labels.
    :param y: array of classifications in the same order of x.
    :param nmin: minimum number of observations (elements) in order to create a Node.
    :param minleaf: minimum number of observations (elements) in order to create a leaf.
    :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
    :return: Tree created using the training set.
    """

    tree = Tree()
    tree.grow(x, y, nmin, minleaf, nfeat)
    return tree


def tree_pred(data: [[]], tree: Tree):
    """
    This function uses a single Tree to classify the data received in input.

    :param data: data to be classified.
    :param tree: tree used to classify the data in input.
    :return: classification of the data.
    """

    return tree.classify(data)


def tree_grow_b(x: [[]], y: [], nmin: int, minleaf: int, nfeat: int, m: int):
    """
    This function creates a Forest object tailored for the given data.
    It can be used also to let grow a Bagging.
    A Forest is a List of Trees.

    :param x: array of data attributes and values passed without the classification labels.
    :param y: array of classifications in the same order of x.
    :param nmin: minimum number of observations (elements) in order to create a Node.
    :param minleaf: minimum number of observations (elements) in order to create a leaf.
    :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
    :param m: number of trees to grow in the current forest.
    :return: Forest created using the training set.
    """

    forest = Forest()
    trees = forest.grow_b(x, y, nmin, minleaf, nfeat, m)
    return trees


def tree_pred_b(data: [[]], trees: [Tree]):
    """
    This function uses a Forest (or Bagging) to classify the data received in input.

    :param data: data to be classified.
    :param trees: list of Trees used to classify the data.
    :return: classification of the data.
    """

    forest = Forest()
    return forest.pred_b(data, trees)


# ---------------------------------------------------------------------------------------------


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


def menu():
    """
    This function implement a command-line menu to make the application interactive for a user.
    """

    possible_choices = ["0", "1", "2", "3", "4", "5", "6"]
    print("\nPress 1 to construct and evaluate a single tree")
    print("Press 2 to construct and evaluate a random forest")
    print("Press 3 to construct and evaluate a bagging")
    print("Press 4 to run a McNemar test to know the statistical significance in accuracy between models")
    print("Press 5 to print the current single tree")
    print("Press 6 to change the nmin, minleaf and m parameters")
    print("Press 0 to exit")
    choice = input("Choice: ")

    while choice not in possible_choices:
        print("\nWrong input! Please, insert a valid one.")
        choice = input("Choice: ")

    return int(choice)


def get_params(nmin=15, minleaf=5, m=100):
    """
    Function to change at the beginning of the program the nmin, minleaf and m parameters, if needed.
    :param nmin: new value for the nmin parameter.
    :param minleaf: new value for the minleaf parameter.
    :param m: new value for the m parameter.
    :return: the new nmin, minleaf and m parameters.
    """

    possible_choices = ["0", "1"]
    print(f"\nCurrent parameters are: nmin = {nmin}, minleaf = {minleaf}, m = {m}")
    if nmin == 15 and minleaf == 5 and m == 100:
        print("Those parameters are the ones to use in the eclipse data.")
    print("The nfeat parameter is calculated automatically.")
    print("\nDo you want to insert different parameters for nmin, minleaf and m? (0 for no / 1 for yes)")
    choice = input("Choice: ")

    while choice not in possible_choices:
        print("\nWrong input! Please, insert a valid one.")
        print("Do you want to insert different parameters for nmin, minleaf and m? (0 for no / 1 for yes)")
        choice = input("Choice: ")

    choice = int(choice)

    if choice == 0:

        return nmin, minleaf, m

    else:

        nmin = int(input("\nPlease enter the nmin parameter: "))
        minleaf = int(input("Please enter the minleaf parameter: "))
        m = int(input("Please enter the m parameter: "))

    return nmin, minleaf, m


def statistic_test(result_tree, result_forest, result_bagging, y_test):
    """
    Function to perform a statistic test to see whether the models are similar, given alpha = 0.05.
    We will perform a mcnemar test, hence:
    H0: the models are similar.
    H1: the models are not similar.

    :param result_tree: the array of predicted values given in output by the tree model.
    :param result_forest: the array of predicted values given in output by the random forest model.
    :param result_bagging: the array of predicted values given in output by the bagging model.
    :param y_test: the array of true labels that should have been predicted.
    """

    # McNemar wants a list of True or False based on the fact that a case has been correctly classified
    correct_tree = result_tree == y_test
    correct_random_forest = result_forest == y_test
    correct_bagging = result_bagging == y_test

    alpha = 0.05

    print('\nTree vs Forest...')
    result1 = mcnemar_test(correct_tree, correct_random_forest)
    print('statistic=%.6f' % result1.statistic)
    print('p-value=%.6f' % result1.pvalue)
    if result1.pvalue > alpha:
        print('Fail to reject H0: thus, the models are similar.')
    else:
        print('Reject H0: thus, the models are not similar.')

    print('\nTree vs Bagging...')
    result2 = mcnemar_test(correct_tree, correct_bagging)
    print('statistic=%.6f' % result2.statistic)
    print('p-value=%.6f' % result2.pvalue)
    if result2.pvalue > alpha:
        print('Fail to reject H0: thus, the models are similar.')
    else:
        print('Reject H0: thus, the models are not similar.')

    print('\nForest vs Bagging...')
    result3 = mcnemar_test(correct_random_forest, correct_bagging)
    print('statistic=%.6f' % result3.statistic)
    print('p-value=%.6f' % result3.pvalue)
    if result3.pvalue > alpha:
        print('Fail to reject H0: thus, the models are similar.')
    else:
        print('Reject H0: thus, the models are not similar.')


def mcnemar_test(correct_1: [], correct_2: []):
    """
    In this function, the actual mcnemar test will be performed to compare the differences in accuracy
    given in ouyput by two different models, and understand whether there is a statistical significance difference.

    :param correct_1: array of bools containing True if the i-th case is classified
    correctly by the first model and False otherwise.
    :param correct_2: array of bools containing True if the i-th case is classified
    correctly by the second model algorithm and False otherwise.
    :return: the result object given in output by the mcnemar function.
    """

    true_true, true_false, false_true, false_false = 0, 0, 0, 0

    for i in range(len(correct_1)):
        if correct_1[i]:
            if correct_2[i]:
                true_true += 1
            else:
                true_false += 1
        else:
            if correct_2[i]:
                false_true += 1
            else:
                false_false += 1

    contingency_table = [[true_true, true_false], [false_true, false_false]]

    print("Contingency table:", contingency_table)

    result = mcnemar(contingency_table, exact=False, correction=True)

    return result


# ---------------------------------------------------------------------------------------------


def main():
    """
    This function is the main function of our program.
    """

    print("\n-------------------- Data Mining assignment 1 --------------------")
    print("Code developed by: Di Grandi Daniele, Hartkamp Jens, Hartog Alice.")
    print("------------------------------------------------------------------")

    print("\nInsert the complete path of the training data")
    path_data_train = input("Path: ")

    print("\nInsert the complete path of the testing data")
    path_data_test = input("Path: ")

    print("\nLoading the data...")

    data_train, data_test = read_data(path_data_train, path_data_test)
    x_train, y_train = split_label(data_train, path_data_train)
    x_test, y_test = split_label(data_test, path_data_test)

    # get the attributes in a dictionary form (see the get_attributes_list for a better explanation)
    global attributes
    attributes = get_attributes_list(x_train, data_train)

    # initialize the nmin, minleaf, m and nfeat parameters
    nmin, minleaf, m = get_params()
    nfeat = int(len(x_train[0]))
    nfeat_forest = int(round(sqrt(len(x_train[0]))))

    choice = menu()

    # check if model exists before make a statistical test
    tree_exists = False
    forest_exists = False
    bagging_exists = False

    # rf_mode: whether the random forest mode is active. 0 means no, 1 means that the computation will be performed
    # using the random forest method. This is because in this case, in the split method of a Node,
    # we can select nfeat attributes each time we are going to make a split.
    global rf_mode

    while choice != 0:

        if choice == 1:
            # Tree
            rf_mode = 0
            print("\nConstructing the single tree...")
            t0 = time()
            tree = tree_grow(x_train, y_train, nmin=nmin, minleaf=minleaf, nfeat=nfeat)
            t1 = time()
            tree_exists = True
            print("\nSingle tree has been constructed in", t1 - t0, "seconds.")
            print("\nEvaluating the single tree...")
            t0 = time()
            result_tree = tree_pred(x_test, tree)
            t1 = time()
            evaluation(y_test, result_tree)
            print("\nPredictions for the single tree have been made in", t1 - t0, "seconds.")

        if choice == 2:
            # Random Forest
            rf_mode = 1
            print("\nConstructing the random forest...")
            t0 = time()
            forest = tree_grow_b(x_train, y_train, nmin=nmin, minleaf=minleaf, nfeat=nfeat_forest, m=m)
            t1 = time()
            forest_exists = True
            print("\nRandom forest has been constructed in", t1 - t0, "seconds.")
            print("\nEvaluating the random forest...")
            t0 = time()
            result_forest = tree_pred_b(x_test, forest)
            t1 = time()
            evaluation(y_test, result_forest)
            print("\nPredictions for the random forest have been made in", t1 - t0, "seconds.")

        if choice == 3:
            # Bagging
            rf_mode = 0
            print("\nConstructing the bagging...")
            t0 = time()
            bagging = tree_grow_b(x_train, y_train, nmin=nmin, minleaf=minleaf, nfeat=nfeat, m=m)
            t1 = time()
            bagging_exists = True
            print("\nBagging has been constructed in", t1 - t0, "seconds.")
            print("\nEvaluating the bagging...")
            t0 = time()
            result_bagging = tree_pred_b(x_test, bagging)
            t1 = time()
            evaluation(y_test, result_bagging)
            print("\nPredictions for bagging have been made in", t1 - t0, "seconds.")

        if choice == 4:
            # Perform a statistical test between the 3 models, if they exists
            if tree_exists and forest_exists and bagging_exists:
                statistic_test(result_tree, result_forest, result_bagging, y_test)

            else:
                print("\nError! One or more models don't exist!")
                print("Please, call this function after having constructed all three models.")

        if choice == 5:
            # Visualize the Tree
            if tree_exists:
                print("\nTree:")
                print_tree(node=tree.root, attributes=attributes)
            else:
                print("\nError! A tree must have been created first, in order to be printed.")

        if choice == 6:
            # Change the nmin, minleaf and m parameters
            nmin, minleaf, m = get_params(nmin, minleaf, m)

        choice = menu()


# ---------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
