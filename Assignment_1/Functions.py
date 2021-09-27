from Assignment_1.Structures import Tree


def tree_grow(x, y, nmin, minleaf, nfeat, attributes):
    """
    This function creates a Tree object tailored for the given data.

    :param x: array of data attributes and values passed without the classification labels.
    :param y: array of classifications in the same order of x.
    :param nmin: minimum number of observations (elements) in order to create a Node.
    :param minleaf: minimum number of observations (elements) in order to create a leaf.
    :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
    :param attributes: a dictionary that specifies the attributes available to make a split.
    Format: attributes = {0: "attribute_0", 1: "attribute_1", ...}.
    :return: Tree created using the training set.
    """

    tree = Tree()
    tree.grow(x, y, nmin, minleaf, nfeat, attributes)
    return tree


def tree_pred(data: [[]], tree: Tree):
    """
    This function classifies the data received in input.

    :param data: data to be classified.
    :param tree: tree used to classify the data in input.
    :return: classification of the data.
    """

    return tree.classify(data)


# create tree_grow_b()



# create tree_pred_b()
