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

- for each model, make a unigram and unigram + bigram version

- 10 folds cross validation (or out-of-bag validation, for random forest) for parameter tuning, minimal parameters to tune:
        multinomial naive bayes --> number of features k that have to be selected (result of feature selection)
        regularized logistic regression --> lambda
        classification tree --> cost-complexity pruning parameter alpha
        random forest --> number of trees m, number of randomly selected features nfeat

- calculate different metrics: accuracy, precision recall and f1 score

- perform a mcnemar test on accuracy for the 8 models to answer 5 questions:

1. How does the performance of the generative linear model (multinomial naive Bayes) compare to the discriminative
linear model (regularized logistic regression)?
2. Is the random forest able to improve on the performance of the linear classifiers?
3. Does performance improve by adding bigrams features, instead of using just unigrams?
4. What are the five most important terms (features) pointing towards a fake review?
5. What are the five most important terms (features) pointing towards a genuine review?


Path: "/Users/danieledigrandi/Desktop/University/UU/Data Mining/DataMining_2021/Assignment_2/data/op_spam_v1.4/negative_polarity"
"""

from Assignment_2.utils import *
from Assignment_2.Preprocessing import *
from Assignment_2.Data_Analysis import *
from Assignment_2.Feature_Selection import *
from Assignment_2.ML_Algorithms import *

from sklearn.metrics import RocCurveDisplay
from sklearn import tree

# if you want to use the program, make sure to change the path folder!!

def main():

    path = f"/Users/danieledigrandi/Desktop/University/UU/Data Mining/DataMining_2021/Assignment_2/data/op_spam_v1.4/negative_polarity"

    overall_unigrams_train, overall_unigrams_test, overall_bigrams_train, overall_bigrams_test, unigrams_train, bigrams_train, unigrams_test, bigrams_test = preprocess_data(path)

    y_train, y_test = read_labels(path)

    # ------------------------------------------------------------------------
    # only for naive bayes:
    # feature selection by eliminating sparse words and with entropy (mutual information...)
    feature_selection = False

    print("Extracting the features...")

    if feature_selection:

        multinomial_bayes_tuning(unigrams_train, bigrams_train, overall_unigrams_train, overall_bigrams_train, y_train)
        exit()

    else:

        unigrams_sparse_threshold, unigrams_mutual_threshold, bigrams_sparse_threshold, bigrams_mutual_threshold = get_best_params('NB')

        not_sparsed_unigrams = eliminate_features(overall_unigrams_train, unigrams_sparse_threshold)
        not_sparsed_bigrams = eliminate_features(overall_bigrams_train, bigrams_sparse_threshold)

        mutual_info_unigrams = mutual_information(unigrams_train, overall_unigrams_train, y_train)
        mi_unigrams_bayes = eliminate_features(mutual_info_unigrams, unigrams_mutual_threshold)

        mutual_info_bigrams = mutual_information(bigrams_train, overall_bigrams_train, y_train)
        mi_bigrams_bayes = eliminate_features(mutual_info_bigrams, bigrams_mutual_threshold)

        overall_unigrams_train_bayes = merge_common_features(not_sparsed_unigrams, mi_unigrams_bayes)
        overall_bigrams_train_bayes = merge_common_features(not_sparsed_bigrams, mi_bigrams_bayes)

    # ------------------------------------------------------------------------
    # exploratory analysis of the data

    # perform_data_analysis(overall_unigrams_train)

    # perform_data_analysis(overall_bigrams_train)

    # most_frequent(mutual_info_unigrams, values_to_print=20)
    # most_frequent(overall_unigrams_train, values_to_print=20)

    # most_frequent(mutual_info_bigrams, values_to_print=20)
    # most_frequent(overall_bigrams_train, values_to_print=20)

    # ------------------------------------------------------------------------
    # definitive features extraction

    general_unigrams_dictionary = list(overall_unigrams_train.keys())
    general_bigrams_dictionary = list(overall_bigrams_train.keys())
    general_unigram_dictionary_bayes = list(overall_unigrams_train_bayes.keys())
    general_bigrams_dictionary_bayes = list(overall_bigrams_train_bayes.keys())

    unigrams_x_train, bigrams_x_train = extract_features_train(unigrams_train, bigrams_train, overall_unigrams_train, overall_bigrams_train)
    unigrams_x_test, bigrams_x_test = extract_features_test(unigrams_test, bigrams_test, general_unigrams_dictionary, general_bigrams_dictionary)

    unigrams_x_bayes_train, bigrams_x_bayes_train = extract_features_train(unigrams_train, bigrams_train, overall_unigrams_train_bayes, overall_bigrams_train_bayes)
    unigrams_x_bayes_test, bigrams_x_bayes_test = extract_features_test(unigrams_test, bigrams_test, general_unigram_dictionary_bayes, general_bigrams_dictionary_bayes)

    # ------------------------------------------------------------------------
    # analysis with the models

    alpha_st_unigrams, alpha_st_bigrams = get_best_params('ST')
    c_unigrams, c_bigrams = get_best_params('LR')
    m_unigrams, m_bigrams, nfeat_unigrams, nfeat_bigrams = get_best_params('RF')


    # unigrams
    single_tree_u = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', ccp_alpha=alpha_st_unigrams)
    single_tree_u.fit(unigrams_x_train, y_train)
    y_st_unigrams_pred = single_tree_u.predict(unigrams_x_test)
    print("\nUNIGRAMS - ST")
    evaluation(y_test, y_st_unigrams_pred)

    logistic_u = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l1', solver='saga', C=c_unigrams)
    logistic_u.fit(unigrams_x_train, y_train)
    y_logistic_unigrams_pred = logistic_u.predict(unigrams_x_test)
    print("\nUNIGRAMS - LR")
    evaluation(y_test, y_logistic_unigrams_pred)

    random_forest_u = RandomForestClassifier(random_state=0, criterion='gini', bootstrap=True, n_estimators=m_unigrams, max_features=nfeat_unigrams)
    random_forest_u.fit(unigrams_x_train, y_train)
    y_rf_unigrams_pred = random_forest_u.predict(unigrams_x_test)
    print("\nUNIGRAMS - RF")
    evaluation(y_test, y_rf_unigrams_pred)

    naive_bayes_u = MultinomialNB()
    naive_bayes_u.fit(unigrams_x_bayes_train, y_train)
    y_nb_unigrams_pred = naive_bayes_u.predict(unigrams_x_bayes_test)
    print("\nUNIGRAMS - NB")
    evaluation(y_test, y_nb_unigrams_pred)


    # bigrams
    single_tree_b = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', ccp_alpha=alpha_st_bigrams)
    single_tree_b.fit(bigrams_x_train, y_train)
    y_st_bigrams_pred = single_tree_b.predict(bigrams_x_test)
    print("\nBIGRAMS - ST")
    evaluation(y_test, y_st_bigrams_pred)

    logistic_b = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l1', solver='saga', C=c_bigrams)
    logistic_b.fit(bigrams_x_train, y_train)
    y_logistic_bigrams_pred = logistic_b.predict(bigrams_x_test)
    print("\nBIGRAMS - LR")
    evaluation(y_test, y_logistic_bigrams_pred)

    random_forest_b = RandomForestClassifier(random_state=0, criterion='gini', bootstrap=True, n_estimators=m_bigrams, max_features=nfeat_bigrams)
    random_forest_b.fit(bigrams_x_train, y_train)
    y_rf_bigrams_pred = random_forest_b.predict(bigrams_x_test)
    print("\nBIGRAMS - RF")
    evaluation(y_test, y_rf_bigrams_pred)

    naive_bayes_b = MultinomialNB()
    naive_bayes_b.fit(bigrams_x_bayes_train, y_train)
    y_nb_bigrams_pred = naive_bayes_b.predict(bigrams_x_bayes_test)
    print("\nBIGRAMS - NB")
    evaluation(y_test, y_nb_bigrams_pred)

    # statistical tests
    test = False

    if test:

        # QUESTION 1
        print("\nQUESTION 1")
        statistic_test(y_nb_unigrams_pred, y_logistic_unigrams_pred, y_test, "MNB-unigrams", "LR-unigrams")
        statistic_test(y_nb_bigrams_pred, y_logistic_bigrams_pred, y_test, "ST-bigrams", "LR-bigrams")

        # QUESTION 2
        print("\nQUESTION 2")
        statistic_test(y_rf_unigrams_pred, y_logistic_unigrams_pred, y_test, "RF-unigrams", "LR-unigrams")
        statistic_test(y_rf_unigrams_pred, y_nb_unigrams_pred, y_test, "RF-unigrams", "MNB-unigrams")
        statistic_test(y_rf_bigrams_pred, y_logistic_bigrams_pred, y_test, "RF-bigrams", "LR-bigrams")
        statistic_test(y_rf_bigrams_pred, y_nb_bigrams_pred, y_test, "RF-bigrams", "MNB-bigrams")

        # QUESTION 3
        print("\nQUESTION 3")
        statistic_test(y_st_unigrams_pred, y_st_bigrams_pred, y_test, "ST-unigrams", "ST-bigrams")
        statistic_test(y_logistic_unigrams_pred, y_logistic_bigrams_pred, y_test, "LR-unigrams", "LR-bigrams")
        statistic_test(y_rf_unigrams_pred, y_rf_bigrams_pred, y_test, "RF-unigrams", "RF-bigrams")
        statistic_test(y_nb_unigrams_pred, y_nb_bigrams_pred, y_test, "NB-unigrams", "NB-bigrams")

        # OVERALL UNIGRAMS
        print("\nOVERALL UNIGRAMS")
        statistic_test(y_st_unigrams_pred, y_logistic_unigrams_pred, y_test, "ST-unigrams", "LR-unigrams")
        statistic_test(y_st_unigrams_pred, y_rf_unigrams_pred, y_test, "ST-unigrams", "RF-unigrams")
        statistic_test(y_st_unigrams_pred, y_nb_unigrams_pred, y_test, "ST-unigrams", "MNB-unigrams")

        statistic_test(y_logistic_unigrams_pred, y_rf_unigrams_pred, y_test, "LR-unigrams", "RF-unigrams")
        statistic_test(y_logistic_unigrams_pred, y_nb_unigrams_pred, y_test, "LR-unigrams", "MNB-unigrams")

        statistic_test(y_rf_unigrams_pred, y_nb_unigrams_pred, y_test, "RF-unigrams", "NB-unigrams")

        # OVERALL BIGRAMS
        print("\nOVERALL BIGRAMS")
        statistic_test(y_st_bigrams_pred, y_logistic_bigrams_pred, y_test, "ST-bigrams", "LR-bigrams")
        statistic_test(y_st_bigrams_pred, y_rf_bigrams_pred, y_test, "ST-bigrams", "RF-bigrams")
        statistic_test(y_st_bigrams_pred, y_nb_bigrams_pred, y_test, "ST-bigrams", "MNB-bigrams")

        statistic_test(y_logistic_bigrams_pred, y_rf_bigrams_pred, y_test, "LR-bigrams", "RF-bigrams")
        statistic_test(y_logistic_bigrams_pred, y_nb_bigrams_pred, y_test, "LR-bigrams", "MNB-bigrams")

        statistic_test(y_rf_bigrams_pred, y_nb_bigrams_pred, y_test, "RF-bigrams", "NB-bigrams")

    # ROC evaluation
    ROC = False
    
    if ROC:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
    
        RocCurveDisplay.from_estimator(single_tree_b, bigrams_x_test, y_test, name="ST", ax=ax1)
        RocCurveDisplay.from_estimator(logistic_b, bigrams_x_test, y_test, name="LR", ax=ax1)
        RocCurveDisplay.from_estimator(random_forest_b, bigrams_x_test, y_test, name="RF", ax=ax1)
        RocCurveDisplay.from_estimator(naive_bayes_b, bigrams_x_bayes_test, y_test, name="MNB", ax=ax1)
    
        plt.title("ROC and AUC for unigrams + bigrams classifiers")
    
        plt.show()

    # show the most important features pointing towards a deceptive/truthful review
    show_most_important_features_MNB(naive_bayes_u, general_unigram_dictionary_bayes)
    show_most_important_features_MNB(naive_bayes_b, general_bigrams_dictionary_bayes)

    # plot the single tree structure
    plot_tree = False
    if plot_tree:
        fig = plt.figure(figsize=(15,10))
        tree.plot_tree(single_tree_b,
                       feature_names=general_bigrams_dictionary,
                       class_names=["deceptive", "truthful"],
                       fontsize=12,
                       filled=True)

        fig.savefig("decision_tree.png", bbox_inches="tight", dpi=300)
        plt.show()


if __name__ == '__main__':
    main()
