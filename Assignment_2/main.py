# Data mining project 2021 - assignment 2


"""
TO-DO

- reading and storing the data (os.walk)

- data preprocessing (e.g. stripping punctuation, removing stop-words, stemming, etc.)

- dividing the data in training (folders 1-4, 640 cases) and test (folder 5, 160 cases)

- only for naive bayes model, feature selection (remove sparse data, use entropy: mutual information)

- setting up 4 models:
        multinomial naive bayes
        regularized logistic regression
        classification tree
        random forest

- for each model, make a unigram and bigram version

- 10 folds cross validation (or out-of-bag validation, for random forest) for parameter tuning, parameters to tune:
        multinomial naive bayes --> number of features k that have to be selected
        regularized logistic regression --> lambda
        classification tree --> cost-complexity pruning parameter alpha
        random forest --> number of trees, number of randomly selected features

- calculate different metrics: accuracy, precision recall and f1 score

- perform a cross validated paired ttest5x2 on accuracy for the 8 models
(this time is possible, since we use scikit-learn for the models)

- address 5 questions:

1. How does the performance of the generative linear model (multinomial naive Bayes) compare to the discriminative
linear model (regularized logistic regression)?
2. Is the random forest able to improve on the performance of the linear classifiers?
3. Does performance improve by adding bigram features, instead of using just unigrams?
4. What are the five most important terms (features) pointing towards a fake review?
5. What are the five most important terms (features) pointing towards a genuine review?



Path: "/Users/danieledigrandi/Desktop/University/UU/Data Mining/DataMining_2021/Assignment_2/data/op_spam_v1.4/negative_polarity"

"""

from Assignment_2.utils import *
from Assignment_2.Preprocessing import *
from Assignment_2.Data_Analysis import *
from Assignment_2.Feature_Selection import *
from Assignment_2.ML_Algorithms import *



def main():

    overall_unigrams_train, overall_unigrams_test, overall_bigrams_train, overall_bigrams_test, unigrams_train, bigrams_train, unigrams_test, bigrams_test = preprocess_data("/Users/danieledigrandi/Desktop/"
                                                                                 "University/UU/Data Mining/DataMining_2021/Assignment_2/data"
                                                                                 "/op_spam_v1.4/negative_polarity")

    y_train, y_test = read_labels("/Users/danieledigrandi/Desktop/University/UU/Data Mining/DataMining_2021/Assignment_2/data/op_spam_v1.4/negative_polarity")

    # ------------------------------------------------------------------------
    # only for naive bayes:
    # feature selection by eliminating sparse words and with entropy (mutual information...)

    feature_selection = False

    if feature_selection:
        # thresholds for eliminating sparse words
        unigrams_sparse_threshold = 1  # to be tuned...
        bigrams_sparse_threshold = 1  # to be tuned...

        not_sparsed_unigrams, not_sparsed_bigrams = eliminate_features(overall_unigrams_train, overall_bigrams_train, unigrams_sparse_threshold, bigrams_sparse_threshold)

        # thresholds for eliminating words with low mutual information
        unigrams_mi_threshold = 0  # to be tuned...
        bigrams_mi_threshold = 0  # to be tuned...

        print("Computing the unigrams mutual information...")
        mutual_info_unigrams = mutual_information(unigrams_train, overall_unigrams_train, y_train)
        print("Computing the bigrams mutual information...")
        mutual_info_bigrams = mutual_information(bigrams_train, overall_bigrams_train, y_train)

        most_frequent(mutual_info_unigrams)
        most_frequent(mutual_info_bigrams)

        mi_unigrams_bayes, mi_bigrams_bayes = eliminate_features(mutual_info_unigrams, mutual_info_bigrams, unigrams_mi_threshold, bigrams_mi_threshold)

        overall_unigrams_train_bayes, overall_bigrams_train_bayes = merge_common_features(not_sparsed_unigrams, not_sparsed_bigrams, mi_unigrams_bayes, mi_bigrams_bayes)

        print("Length of original unigrams dictionary:", len(overall_unigrams_train))
        print("Length of non-sparsed unigrams dictionary:", len(not_sparsed_unigrams))
        print("Length of the unigrams mutual-information dictionary:", len(mi_unigrams_bayes))
        print("Length of the unigrams merged dictionary (non-sparsed & mutual-information):", len(overall_unigrams_train_bayes))
        print("\n")
        print("Length of original bigrams dictionary:", len(overall_bigrams_train))
        print("Length of non-sparsed bigrams dictionary:", len(not_sparsed_bigrams))
        print("Length of the bigrams mutual-information dictionary:", len(mi_bigrams_bayes))
        print("Length of the bigrams merged dictionary (non-sparsed & mutual-information):", len(overall_bigrams_train_bayes))

    else:
        overall_unigrams_train_bayes = overall_unigrams_train
        overall_bigrams_train_bayes = overall_bigrams_train

    # ------------------------------------------------------------------------
    # exploratory analysis of the data

    #perform_data_analysis(overall_unigrams_train)
    perform_data_analysis(overall_bigrams_train)

    # ------------------------------------------------------------------------
    # definitive features extraction

    general_unigrams_dictionary = list(overall_unigrams_train.keys())
    general_bigrams_dictionary = list(overall_bigrams_train.keys())
    general_unigram_dictionary_bayes = list(overall_unigrams_train_bayes.keys())
    general_bigrams_dictionary_bayes = list(overall_bigrams_train_bayes.keys())

    print("Extracting the features...")

    unigrams_x_train, bigrams_x_train = extract_features_train(unigrams_train, bigrams_train, overall_unigrams_train, overall_bigrams_train)
    unigrams_x_test, bigrams_x_test = extract_features_test(unigrams_test, bigrams_test, general_unigrams_dictionary, general_bigrams_dictionary)

    unigrams_x_bayes_train, bigrams_x_bayes_train = extract_features_train(unigrams_train, bigrams_train, overall_unigrams_train_bayes, overall_bigrams_train_bayes)
    unigrams_x_bayes_test, bigrams_x_bayes_test = extract_features_test(unigrams_test, bigrams_test, general_unigram_dictionary_bayes, general_bigrams_dictionary_bayes)

    # ------------------------------------------------------------------------
    # analysis with the models

    random_forest_tuning(unigrams_x_train, y_train, unigrams_x_test, y_test)



if __name__ == '__main__':
    main()
