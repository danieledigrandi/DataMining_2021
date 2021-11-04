from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import l1_min_c
from Assignment_2.utils import *
from Assignment_2.Feature_Selection import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def logistic_regression_tuning(X_train, y_train):

    lambda_scores = []
    lambda_number_of_remained_features = []
    C_range = l1_min_c(X_train, y_train, loss="log") * np.logspace(0, 7, 16)  # formula taken here: https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#sphx-glr-auto-examples-linear-model-plot-logistic-path-py
    # C_range = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 120, 160, 200, 220, 260, 300, 320, 360, 400, 420, 460, 500, 520, 560, 600, 1000, 1500, 2000, 3000, 4000]

    for C in C_range:
        print('Testing C:', C, "(lambda:", str(1/C) + ")")
        model = LogisticRegression(random_state=0, penalty='l1', solver='saga', C=C)
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        lambda_scores.append(scores.mean())

        model = LogisticRegression(random_state=0, penalty='l1', solver='saga', C=C).fit(X_train, y_train)

        y = list(model.coef_)
        zeros = 0

        num_of_original_features = len(X_train[0])

        for i in y:
            for j in range(len(i) - 1):
                if i[j] == 0:
                    zeros += 1

        lambda_number_of_remained_features.append(num_of_original_features - zeros)

    lambda_range = [1/i for i in C_range]
    log_lambda_range = [int(np.log(i)*1000)/1000 for i in lambda_range]

    print('Range of C:', C_range)
    print('Corresponding lambdas:', lambda_range)
    print('Accuracies:', lambda_scores)
    print('Remained features:', lambda_number_of_remained_features)

    df = pd.DataFrame()

    df['C_range'] = C_range
    df['lambda_range'] = lambda_range
    df['log_lambda_range'] = log_lambda_range
    df['lambda_number_of_remained_features'] = lambda_number_of_remained_features
    df['lambda_scores'] = lambda_scores

    filename = 'Logistic_regression_CV.csv'

    df.to_csv(filename, index=False)


def single_tree_tuning(X_train, y_train, plot_impurity=False):
    # approach partially based on: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

    alpha_scores = []
    node_counts = []
    depth = []
    n_leafs = []

    model = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best')
    path = model.cost_complexity_pruning_path(X_train, y_train)

    """
    Minimal cost complexity pruning recursively finds the node with the “weakest link”. 
    The weakest link is characterized by an effective alpha, where the nodes with the smallest effective alpha are pruned first. 
    To get an idea of what values of ccp_alpha could be appropriate, scikit-learn provides DecisionTreeClassifier.cost_complexity_pruning_path 
    that returns the effective alphas and the corresponding total leaf impurities at each step of the pruning process. 
    As alpha increases, more of the tree is pruned, which increases the total impurity of its leaves.
    """

    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    if plot_impurity:
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], drawstyle="steps-post")
        ax.set_xlabel("Effective alpha")
        ax.set_ylabel("Total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set (bigrams, ST)")
        plt.show()

    for alpha in ccp_alphas:
        print('Testing alpha:', alpha)
        model = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', ccp_alpha=alpha)
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        alpha_scores.append(scores.mean())

        model = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', ccp_alpha=alpha).fit(X_train, y_train)

        node_counts.append(model.tree_.node_count)
        depth.append(model.tree_.max_depth)
        n_leafs.append(model.tree_.n_leaves)

    print('Range of alpha:', ccp_alphas)
    print('Accuracies:', alpha_scores)
    print('Depth:', depth)
    print('Node counts:', node_counts)
    print('Leaf counts:', n_leafs)

    df = pd.DataFrame()

    df['Range_of_alpha'] = ccp_alphas
    df['Accuracies'] = alpha_scores
    df['Depth'] = depth
    df['Node counts'] = node_counts
    df['Leaf counts'] = n_leafs

    filename = 'Single_tree_CV.csv'

    df.to_csv(filename, index=False)


# Multinomial naive bayes
def multinomial_bayes_tuning(unigrams_train, bigrams_train, overall_unigrams_train, overall_bigrams_train, y_train):

    # thresholds for eliminating sparse words
    unigrams_sparse_threshold = [1, 2, 6, 10, 15, 20, 30, 50, 60, 80, 100, 200, 300]
    bigrams_sparse_threshold = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 70, 100]

    # thresholds for eliminating words with low mutual information
    unigrams_mi_threshold = [0, 0.0001, 0.0004, 0.001, 0.0015, 0.002, 0.004, 0.01, 0.02, 0.04]
    bigrams_mi_threshold = [0, 0.0003, 0.001, 0.0014, 0.0016, 0.005, 0.01, 0.013, 0.015, 0.02]

    df_unigrams = pd.DataFrame(columns=['t_sparse', 't_mutual', 'length_original_dictionary', 'length_after_sparse', 'length_after_mutual', 'length_merged', 'accuracy'])
    df_bigrams = pd.DataFrame(columns=['t_sparse', 't_mutual', 'length_original_dictionary', 'length_after_sparse', 'length_after_mutual', 'length_merged', 'length_unigrams', 'tot_bigrams_feature', 'accuracy'])


    # unigrams tuning
    for ts in unigrams_sparse_threshold:
        not_sparsed_unigrams = eliminate_features(overall_unigrams_train, ts)

        for tmi in unigrams_mi_threshold:

            print('UNIGRAMS - Testing sparse:', ts, "and mutual information:", tmi)
            mutual_info_unigrams = mutual_information(unigrams_train, overall_unigrams_train, y_train)
            mi_unigrams_bayes = eliminate_features(mutual_info_unigrams, tmi)

            overall_unigrams_train_bayes = merge_common_features(not_sparsed_unigrams, mi_unigrams_bayes)

            unigrams_x_bayes_train, bigrams_x_useless = extract_features_train(unigrams_train, bigrams_train, overall_unigrams_train_bayes, overall_bigrams_train)

            model = MultinomialNB()
            scores = cross_val_score(model, unigrams_x_bayes_train, y_train, cv=10, scoring='accuracy')
            accuracy = scores.mean()

            row = {'t_sparse': ts, 't_mutual': tmi, 'length_original_dictionary': len(overall_unigrams_train), 'length_after_sparse': len(not_sparsed_unigrams), 'length_after_mutual': len(mi_unigrams_bayes), 'length_merged': len(overall_unigrams_train_bayes), 'accuracy': accuracy}

            df_unigrams = df_unigrams.append(row, ignore_index=True)

    filename = 'Unigrams_MNB_CV.csv'

    df_unigrams.to_csv(filename, index=False)


    # bigrams tuning
    for ts in bigrams_sparse_threshold:
        not_sparsed_unigrams = eliminate_features(overall_unigrams_train, ts)
        not_sparsed_bigrams = eliminate_features(overall_bigrams_train, ts)

        for tmi in bigrams_mi_threshold:

            print('BIGRAMS - Testing sparse:', ts, "and mutual information:", tmi)
            mutual_info_unigrams = mutual_information(unigrams_train, overall_unigrams_train, y_train)
            mi_unigrams_bayes = eliminate_features(mutual_info_unigrams, tmi)

            mutual_info_bigrams = mutual_information(bigrams_train, overall_bigrams_train, y_train)
            mi_bigrams_bayes = eliminate_features(mutual_info_bigrams, tmi)

            overall_unigrams_train_bayes = merge_common_features(not_sparsed_unigrams, mi_unigrams_bayes)
            overall_bigrams_train_bayes = merge_common_features(not_sparsed_bigrams, mi_bigrams_bayes)

            unigrams_x_bayes_train, bigrams_x_bayes_train = extract_features_train(unigrams_train, bigrams_train, overall_unigrams_train_bayes, overall_bigrams_train_bayes)

            model = MultinomialNB()
            scores = cross_val_score(model, bigrams_x_bayes_train, y_train, cv=10, scoring='accuracy')
            accuracy = scores.mean()

            row = {'t_sparse': ts, 't_mutual': tmi, 'length_original_dictionary': len(overall_bigrams_train), 'length_after_sparse': len(not_sparsed_bigrams), 'length_after_mutual': len(mi_bigrams_bayes), 'length_merged': len(overall_bigrams_train_bayes), 'length_unigrams': len(overall_unigrams_train_bayes), 'tot_bigrams_feature': len(overall_bigrams_train_bayes) + len(overall_unigrams_train_bayes), 'accuracy': accuracy}

            df_bigrams = df_bigrams.append(row, ignore_index=True)

    filename = 'Bigrams_MNB_CV.csv'

    df_bigrams.to_csv(filename, index=False)


# Random forest
def random_forest_tuning(X_train, y_train, X_test, y_test):

    parameters = {
        "n_estimators": [30, 50, 70, 90, 100, 110, 120, 130],  # m
        "max_features": ["sqrt", "log2", 0.4, 0.8],  # nfeat
    }

    model = RandomForestClassifier(random_state=0, criterion='gini', bootstrap=True)
    grid = GridSearchCV(model, parameters)
    grid_fit = grid.fit(X_train, y_train)
    print(grid_fit.best_estimator_.get_params())
    print()
    print(grid_fit)

    y_pred = grid_fit.predict(X_train)

    evaluation(y_train, y_pred)
