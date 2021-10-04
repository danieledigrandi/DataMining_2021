# Data mining project 2021 - assignment 1

from Assignment_1.Utils import read_data, split_label, get_attributes_list, print_tree, evaluation
from math import sqrt
from Assignment_1.Functions import *


"""

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

"""


def menu():

    possible_choices = ["0", "1", "2", "3", "4"]
    print("\nPress 1 to construct and evaluate a single tree")
    print("Press 2 to construct and evaluate a random forest")
    print("Press 3 to construct and evaluate a bagging")
    print("Press 4 to print the current single tree")
    print("Press 0 to exit")
    choice = input("Choice: ")

    if choice not in possible_choices:
        print("\nWrong input! Please, insert a valid one.")
        choice = menu()

    return int(choice)


def main():

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

    attributes = get_attributes_list(x_train, data_train)
    original = attributes.copy()

    nmin = 15
    minleaf = 5
    nfeat = int(len(x_train[0]))
    nfeat_forest = int(round(sqrt(len(x_train[0]))))
    m = 100

    choice = menu()

    tree_exists = False

    while choice != 0:

        if choice == 1:
            # Tree
            print("\nConstructing the single tree...")
            tree = tree_grow(x_train, y_train, nmin=nmin, minleaf=minleaf, nfeat=nfeat, attributes=attributes, rf_mode=0)
            tree_exists = True
            print("\nSingle tree has been constructed.")
            print("\nEvaluating the single tree...")
            result_tree = tree_pred(x_test, tree)
            accuracy_tree, precision_tree, recall_tree = evaluation(y_test, result_tree)

        if choice == 2:
            # Random Forest
            print("\nConstructing the random forest...")
            forest = tree_grow_b(x_train, y_train, nmin=nmin, minleaf=minleaf, nfeat=nfeat_forest, m=m, attributes=attributes, rf_mode=1)
            print("\nRandom forest has been constructed.")
            print("\nEvaluating the random forest...")
            result_forest = tree_pred_b(x_test, forest)
            accuracy_forest, precision_forest, recall_forest = evaluation(y_test, result_forest)

        if choice == 3:
            # Bagging
            print("\nConstructing the bagging...")
            bagging = tree_grow_b(x_train, y_train, nmin=nmin, minleaf=minleaf, nfeat=nfeat, m=m, attributes=attributes, rf_mode=0)
            print("\nBagging has been constructed.")
            print("\nEvaluating the bagging...")
            result_bagging = tree_pred_b(x_test, bagging)
            accuracy_bagging, precision_bagging, recall_bagging = evaluation(y_test, result_bagging)

        if choice == 4:
            # Visualize the Tree
            if tree_exists:
                print("\nTree:")
                print_tree(node=tree.root, attributes=original)
            else:
                print("\nError! A tree must have been created first, in order to be printed.")

        choice = menu()


if __name__ == "__main__":
    main()
