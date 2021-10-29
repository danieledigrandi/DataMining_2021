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


def extract_features(unigrams, bigrams, overall_unigrams, overall_bigrams):

    unigrams_x = np.zeros((len(unigrams), len(overall_unigrams)))
    bigrams_x = np.zeros((len(bigrams), len(overall_bigrams)))

    unigrams_words = list(overall_unigrams.keys())
    unigrams_values_files = list(unigrams.values())

    bigrams_words = list(overall_bigrams.keys())
    bigrams_values_files = list(bigrams.values())


    for i in range(len(unigrams_words)):
        for j in range(len(unigrams_values_files)):

            word = unigrams_words[i]

            if word in unigrams_values_files[j]:
                unigrams_x[j][i] = unigrams_values_files[j][word]

    for i in range(len(bigrams_words)):
        for j in range(len(bigrams_values_files)):

            word = bigrams_words[i]

            if word in bigrams_values_files[j]:
                bigrams_x[j][i] = bigrams_values_files[j][word]

    return unigrams_x, bigrams_x


def evaluation(y_test, y_pred):

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred)),
    print("All information of each label:"),
    print(metrics.classification_report(y_test, y_pred)),

    print("Confusion Matrix: \n"),
    print(metrics.confusion_matrix(y_test, y_pred))

