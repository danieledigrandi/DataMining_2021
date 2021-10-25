from Assignment_2.utils import notImplemented
from collections import Counter

def perform_data_analysis(unigrams_train, bigrams_train):

    # statistics for example on how many unigrams are present only once, twice or more
    # statistics for example on how many biagrams are present only once, twice or more

    notImplemented()


def analyse_mutual_info(mutual_info):

    values_to_print = 20

    k = Counter(mutual_info)
    high = k.most_common(values_to_print)

    for i in high:
        print(i[0], "=", i[1])


