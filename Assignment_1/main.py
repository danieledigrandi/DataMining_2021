# Data mining project 2021 - assignment 1
import pandas as pd
import numpy
from Assignment_1.Utils import read_data, split_label, get_attributes_list
from Assignment_1.Structures import Tree
from math import sqrt


def main():

    file_train = "credit"  # possible values: "credit", "pima", "eclipse-train"
    file_test = "credit"  # possible values: "credit", "pima", "eclipse-test"


    data_train = read_data(file_train)
    data_test = read_data(file_test)
    x_train, y_train = split_label(data_train, file_train)
    x_test, y_test = split_label(data_test, file_test)

    attributes = get_attributes_list(x_train, data_train)
    original = attributes.copy()

    print(attributes)

    nmin = 2
    minleaf = 1
    nfeat = int(len(x_train[0]))
    nfeat_forest = int(round(sqrt(len(x_train[0]))))
    m = 0

    print(nfeat, nfeat_forest)

    # Constructing the Tree
    tree = Tree()
    tree.grow(x_train, y_train, nmin=nmin, minleaf=minleaf, nfeat=None, attributes=attributes)

if __name__ == "__main__":
    main()