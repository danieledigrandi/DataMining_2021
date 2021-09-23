# read the data ---Alice

# compute the best split

def compute_best_split(x_data: [], y_data: [], minleaf: int):
    """
    This function calculates whether a split should be made on a node using the gini impurity
    it finds the possible splits and returns the split with the lowest impurity

    :param x_data: array of values to perform a split in
    :param y_data: array of classifications matching the order of data_x
    :param minleaf: minimum number of elements in each part of the split in order for a split to be valid
    :return: value of the split, impurities for both parts of the split, and proportional length of both
    sides of the split
    """
    None