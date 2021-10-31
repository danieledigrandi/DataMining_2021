import os
import numpy as np
from sklearn import metrics


def notImplemented():

    print("Not implemented yet")


def read_labels(path):

    y_train = {}
    y_test = {}

    for root, dirs, files in os.walk(path):

        for file in files:

            if file.endswith(".txt") and file != "trainingLabels.txt" and file != "testLabels.txt":

                if root.endswith("deceptive_from_MTurk/fold5"):

                    y_test[file] = 0

                elif root.endswith("truthful_from_Web/fold5"):

                    y_test[file] = 1

                elif "deceptive_from_MTurk" in root and not root.endswith("fold5"):

                    y_train[file] = 0

                elif "truthful_from_Web" in root and not root.endswith("fold5"):

                    y_train[file] = 1

    y_train_list = list(y_train.values())
    y_test_list = list(y_test.values())

    y_train_np = np.asarray(y_train_list, dtype=int)
    y_test_np = np.asarray(y_test_list, dtype=int)

    return y_train_np, y_test_np


def extract_features_train(unigrams, bigrams, overall_unigrams, overall_bigrams):

    unigrams_x = np.zeros((len(unigrams), len(overall_unigrams)))
    bigrams_x = np.zeros((len(bigrams), len(overall_bigrams) + len(overall_unigrams)))

    unigrams_words = list(overall_unigrams.keys())
    unigrams_values_files = list(unigrams.values())

    bigrams_words = list(overall_bigrams.keys())
    bigrams_values_files = list(bigrams.values())

    for i in range(len(unigrams_words)):
        for j in range(len(unigrams_values_files)):

            word = unigrams_words[i]

            if word in unigrams_values_files[j]:
                unigrams_x[j][i] = unigrams_values_files[j][word]
                bigrams_x[j][i] = unigrams_values_files[j][word]

    for i in range(len(bigrams_words)):
        for j in range(len(bigrams_values_files)):

            word = bigrams_words[i]

            if word in bigrams_values_files[j]:
                bigrams_x[j][len(unigrams_words)+i] = bigrams_values_files[j][word]

    return unigrams_x, bigrams_x


def extract_features_test(unigrams, bigrams, general_unigrams_dictionary, general_bigrams_dictionary):

    unigrams_x = np.zeros((len(unigrams), len(general_unigrams_dictionary)))
    bigrams_x = np.zeros((len(bigrams), len(general_unigrams_dictionary) + len(general_bigrams_dictionary)))

    unigrams_values_files = list(unigrams.values())

    bigrams_values_files = list(bigrams.values())

    for i in range(len(general_unigrams_dictionary)):
        for j in range(len(unigrams_values_files)):

            word = general_unigrams_dictionary[i]

            if word in unigrams_values_files[j]:
                unigrams_x[j][i] = unigrams_values_files[j][word]
                bigrams_x[j][i] = unigrams_values_files[j][word]

    for i in range(len(general_bigrams_dictionary)):
        for j in range(len(bigrams_values_files)):

            word = general_bigrams_dictionary[i]

            if word in bigrams_values_files[j]:
                bigrams_x[j][len(general_unigrams_dictionary)+i] = bigrams_values_files[j][word]

    return unigrams_x, bigrams_x


def evaluation(y_test, y_pred):

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred)),
    print("All information of each label:"),
    print(metrics.classification_report(y_test, y_pred)),

    print("Confusion Matrix: \n"),
    print(metrics.confusion_matrix(y_test, y_pred))


def get_best_params(mode):

    if mode == 'NB':

        unigrams_sparse_threshold = 10
        unigrams_mutual_threshold = 0.0015

        bigrams_sparse_threshold = 1
        bigrams_mutual_threshold = 0.005

        return unigrams_sparse_threshold, unigrams_mutual_threshold, bigrams_sparse_threshold, bigrams_mutual_threshold

    elif mode == 'ST':

        alpha_unigrams = 0.01
        alpha_bigrams = 0.008

        return alpha_unigrams, alpha_bigrams

    elif mode == 'LR':

        c_unigrams = 1.3
        c_bigrams = 0.7

        return c_unigrams, c_bigrams

    elif mode == 'RF':

        m_unigrams = 130
        nfeat_unigrams = "sqrt"

        m_bigrams = 130
        nfeat_bigrams = "sqrt"

        return m_unigrams, m_bigrams, nfeat_unigrams, nfeat_bigrams


