# Data mining project 2021 - assignment 2


"""
TO-DO

- reading and storing the data (os.walk)

- data preprocessing (which type?)

- dividing the data in training (folders 1-4, 640 cases) and test (folder 5, 160 cases)

- only for naive bayes model, feature selection (use entropy)

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

"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import metrics


print("Hello world!")