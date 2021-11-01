from collections import Counter
from statsmodels.stats.contingency_tables import mcnemar


def perform_data_analysis(data):

    # statistics for example on how many unigrams are present only once, twice or more
    # statistics for example on how many biagrams are present only once, twice or more

    present_once = 0
    present_twice = 0

    for key, value in data.items():
        if value == 1:
            present_once += 1
        elif value == 2:
            present_twice += 1

    print(present_once)
    print(present_twice)


def most_frequent(data, values_to_print=20):

    k = Counter(data)
    high = k.most_common(values_to_print)

    for i in high:
        print(i[0], "=", i[1])


def mcnemar_test(correct_1: [], correct_2: []):
    """
    In this function, the actual mcnemar test will be performed to compare the differences in accuracy
    given in ouyput by two different models, and understand whether there is a statistical significance difference.

    :param correct_1: array of bools containing True if the i-th case is classified
    correctly by the first model and False otherwise.
    :param correct_2: array of bools containing True if the i-th case is classified
    correctly by the second model algorithm and False otherwise.
    :return: the result object given in output by the mcnemar function.
    """

    true_true, true_false, false_true, false_false = 0, 0, 0, 0

    for i in range(len(correct_1)):
        if correct_1[i]:
            if correct_2[i]:
                true_true += 1
            else:
                true_false += 1
        else:
            if correct_2[i]:
                false_true += 1
            else:
                false_false += 1

    contingency_table = [[true_true, true_false], [false_true, false_false]]

    print("Contingency table:", contingency_table)

    result = mcnemar(contingency_table, exact=False, correction=True)

    return result


def statistic_test(result_1, result_2, y_test, name_1, name_2):
    """
    Function to perform a statistic test to see whether the models are similar, given alpha = 0.05.
    We will perform a mcnemar test, hence:
    H0: the models are similar.
    H1: the models are not similar.
    """

    # McNemar wants a list of True or False based on the fact that a case has been correctly classified
    correct_1 = result_1 == y_test
    correct_2 = result_2 == y_test

    alpha = 0.05

    print('\n', name_1, 'vs', name_2, '...')
    test = mcnemar_test(correct_1, correct_2)
    print('statistic=%.6f' % test.statistic)
    print('p-value=%.6f' % test.pvalue)
    if test.pvalue > alpha:
        print('Fail to reject H0: thus, the models are similar.')
    else:
        print('Reject H0: thus, the models are not similar.')


def show_most_important_features_MNB(clf, feature_names, n=20):
    """
    The coef_ attribute of MultinomialNB is a re-parameterization of the naive Bayes model as a linear classifier model.
    For a binary classification problem this is basically the log of the estimated probability of a feature given the positive class.
    It means that higher values mean more important features for the positive class.
    """

    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
