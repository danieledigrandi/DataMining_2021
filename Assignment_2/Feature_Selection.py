import numpy as np


def eliminate_features(unigrams, bigrams, unigrams_threshold, bigrams_threshold):

    new_unigrams = unigrams.copy()
    new_bigrams = bigrams.copy()

    # eliminate the features only in the training sample

    for word, value in unigrams.items():
        if value < unigrams_threshold:
            del new_unigrams[word]

    for words, value in bigrams.items():
        if value < bigrams_threshold:
            del new_bigrams[words]

    return new_unigrams, new_bigrams


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


def merge_common_features(not_sparsed_unigrams, not_sparsed_bigrams, mi_unigrams_bayes, mi_bigrams_bayes):

    overall_unigrams_train_bayes = {}
    overall_bigrams_train_bayes = {}

    for key, value in not_sparsed_unigrams.items():
        if key in mi_unigrams_bayes:
            overall_unigrams_train_bayes[key] = value

    for key, value in not_sparsed_bigrams.items():
        if key in mi_bigrams_bayes:
            overall_bigrams_train_bayes[key] = value

    return overall_unigrams_train_bayes, overall_bigrams_train_bayes
