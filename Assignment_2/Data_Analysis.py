from collections import Counter

"""
Implement the function to check if all the texts are different.
"""

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


def most_frequent(data):

    values_to_print = 20

    k = Counter(data)
    high = k.most_common(values_to_print)

    for i in high:
        print(i[0], "=", i[1])
