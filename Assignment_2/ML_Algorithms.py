from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from Assignment_2.utils import evaluation


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
