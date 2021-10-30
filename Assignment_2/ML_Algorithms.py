from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from Assignment_2.utils import evaluation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def logistic_regression_tuning(X_train, y_train):

    lambda_scores = []
    lambda_number_of_remained_features = []
    C_range = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 120, 160, 200, 220, 260, 300, 320, 360, 400, 420, 460, 500, 520, 560, 600, 1000, 1500, 2000, 3000, 4000]

    for C in C_range:
        print('Testing C:', C, "(lambda:", str(1/C) + ")")
        model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l1', solver='saga', C=C)
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        lambda_scores.append(scores.mean())

        model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l1', solver='saga', C=C).fit(X_train, y_train)

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


def plot_CV_graph(parameter_range, remained_features_range, accuracy_range, mode):
    """
    :param mode: 'LR' for logistic regression and 'ST' for single tree.
    For naive bayes and random forest, we cannot plot a 2D graph since for each model we have a combination of
    2 parameters that have to change together and therefore be tuned. The result would be a 3D graph.
    :return:
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xticks(parameter_range)
    ax1.set_yticks(accuracy_range)
    ax1.set_xticklabels(parameter_range, fontsize=8, rotation=90)
    ax1.plot(parameter_range, accuracy_range)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(parameter_range)
    ax2.set_xticklabels(remained_features_range, fontsize=8, rotation=90)

    if mode == 'LR':
        ax1.set_xlabel('Value of LN(lambda) for Logistic Regression model with LASSO penalty')
        ax2.set_xlabel('Number of remained features')
        ax1.set_ylabel('Cross-Validated Accuracy')
        plt.title("Cross-validation: tuning lambda for Logistic Regression")

    if mode == 'ST':
        ax1.set_xlabel('Value of Complexity Pruning alpha for Single Tree model')
        ax2.set_xlabel('Number of remained features')
        ax1.set_ylabel('Cross-Validated Accuracy')
        plt.title("Cross-validation: tuning alpha for Single Tree")

    plt.tight_layout()
    plt.show()


def prova():

    C_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 120, 160, 200, 220, 260, 300, 320, 360, 400, 420, 460, 500, 520, 560, 600, 1000, 1500, 2000, 3000, 4000]
    la = [np.log(1 / i) for i in C_range]
    lambda_range = [int(np.log(1/i)*1000)/1000 for i in C_range]
    print(la)
    print(lambda_range)
    lambda_scores = [i for i in range(48)]
    lambda_number_of_remained_features = [i for i in range(48)]

    fig = plt.figure(figsize=(24, 14), dpi=120)
    ax1 = fig.add_subplot(111)
    ax1.set_xticks(lambda_range)
    ax1.set_yticks(lambda_scores)
    ax1.set_xticklabels(lambda_range, fontsize=6, rotation=90)
    ax1.plot(lambda_range, lambda_scores)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(lambda_range)
    ax2.set_xticklabels(lambda_number_of_remained_features, fontsize=6, rotation=90)

    ax1.set_xlabel('Value of LN(lambda) for Logistic Regression model with LASSO penalty')
    ax2.set_xlabel('Number of remained features')
    ax1.set_ylabel('Cross-Validated Accuracy')
    plt.title("Cross-validation: tuning lambda for Logistic Regression")

    plt.tight_layout()
    plt.show()






# Multinomial naive bayes
def multinomial_bayes(x_train, y_train, x_test, y_test):
    parameters = {
        "alpha": [0.8, 0.9, 1.0, 1.1, 1.2],
        "fit_prior": (True, False),
    }
    clf = MultinomialNB()
    grid = GridSearchCV(clf, parameters)
    grid_fit = grid.fit(x_train, y_train)
    print(grid_fit.get_params())

    y_pred = grid_fit.predict(x_test)

    evaluation(y_test, y_pred)


# Logistic Regression
def logistic_regression(x_train, y_train, x_test, y_test):
    parameters = {
        "penalty": ('l1', 'l2', 'elasticnet', None),
        "dual": (True, False),
        "tol": [0.00005, 0.0001, 0.0002],
        "C": [0.9, 1.0, 1.1], # this
        "fit_intercept": (True, False),
        "solver": ('newton - cg', 'lbfgs', 'liblinear', 'sag', 'saga')
    }
    # Train the model
    clf = LogisticRegression()#max_iter=1000000)
    grid = GridSearchCV(clf, parameters)
    grid_fit = grid.fit(x_train, y_train)
    print(grid_fit.get_params())
    print()
    print(grid_fit.best_estimator_.coef_())
    # Testing the model
    y_pred = grid_fit.predict(x_test)

    evaluation(y_test, y_pred)


# Decision tree
def decision_tree(x_train, y_train, x_test, y_test):
    parameters = {
        "criterion":  ('gini', 'entropy'),
        "splitter": ("best", "random"),
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ("auto", "sqrt", "log2", None),
        "min_impurity_decrease": [0, 0.1, 0.2],
        "ccp_alpha": [0, 0.1, 0.2] # this
    }
    clf = DecisionTreeClassifier()
    grid = GridSearchCV(clf, parameters)
    # Train Decision Tree Classifer
    grid_fit = grid.fit(x_train, y_train)

    print(grid_fit.cv_results_)
    # Predict the response for test dataset
    y_pred = grid_fit.best_estimator_.predict(x_test)

    evaluation(y_test, y_pred)


# Random forest
def random_forest(x_train, y_train, x_test, y_test):
    parameters = {
        "n_estimators": [10, 25, 100], # this
        "criterion": ('gini', 'entropy'),
        "splitter": ("best", "random"),
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ("auto", "sqrt", "log2", None),
        "min_impurity_decrease": [0, 0.1, 0.2],
        "bootstrap": (True, False),
        "max_sample": [None, 50, 100, 250, 0.5, 0.9, 0.2] # this
    }
    clf = RandomForestClassifier()
    grid = GridSearchCV(clf, parameters)
    grid_fit = grid.fit(x_train, y_train)
    print(grid_fit.best_estimator_.get_params())
    print()
    print(grid_fit)

    y_pred = grid_fit.predict(x_test)

    evaluation(y_test, y_pred)
