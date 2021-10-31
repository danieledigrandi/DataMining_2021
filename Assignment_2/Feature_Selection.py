import numpy as np


def eliminate_features(data, data_threshold):

    new_data = data.copy()

    # eliminate the features only in the training sample

    for word, value in data.items():
        if value < data_threshold:
            del new_data[word]

    return new_data


def mutual_information(feature_dictionary, overall_feature_dictionary, label):

    mutual_info = {}

    unigrams_words = list(overall_feature_dictionary.keys())
    unigrams_values_files = list(feature_dictionary.values())
    tot_data = len(unigrams_values_files)

    for i in range(len(unigrams_words)):

        zero_zero = 0
        zero_one = 0
        one_zero = 0
        one_one = 0
        word = unigrams_words[i]

        for j in range(tot_data):

            if word not in unigrams_values_files[j] and label[j] == 0:
                zero_zero += 1

            elif word not in unigrams_values_files[j] and label[j] == 1:
                zero_one += 1

            elif word in unigrams_values_files[j] and label[j] == 0:
                one_zero += 1

            elif word in unigrams_values_files[j] and label[j] == 1:
                one_one += 1

        denom1 = ((zero_zero + zero_one) * (zero_zero + one_zero))
        denom2 = ((zero_zero + zero_one) * (zero_one + one_one))
        denom3 = ((one_zero + one_one) * (zero_zero + one_zero))
        denom4 = ((one_zero + one_one) * (zero_one + one_one))

        if zero_zero != 0:
            first = (zero_zero/tot_data) * np.log2((tot_data * zero_zero)/denom1)
        else:
            first = 0

        if zero_one != 0:
            second = (zero_one/tot_data) * np.log2((tot_data * zero_one)/denom2)
        else:
            second = 0

        if one_zero != 0:
            third = (one_zero/tot_data) * np.log2((tot_data * one_zero)/denom3)
        else:
            third = 0

        if one_one != 0:
            fourth = (one_one/tot_data) * np.log2((tot_data * one_one)/denom4)
        else:
            fourth = 0

        word_information = first + second + third + fourth

        mutual_info[word] = word_information

    return mutual_info


def merge_common_features(not_sparsed, mi_bayes):

    overall_train_bayes = {}

    for key, value in not_sparsed.items():
        if key in mi_bayes:
            overall_train_bayes[key] = value

    return overall_train_bayes
