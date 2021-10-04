from Assignment_1.Structures import Tree, Forest


def tree_grow(x, y, nmin, minleaf, nfeat, attributes, rf_mode):
    """
    This function creates a Tree object tailored for the given data.

    :param x: array of data attributes and values passed without the classification labels.
    :param y: array of classifications in the same order of x.
    :param nmin: minimum number of observations (elements) in order to create a Node.
    :param minleaf: minimum number of observations (elements) in order to create a leaf.
    :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
    :param attributes: a dictionary that specifies the attributes available to make a split.
    Format: attributes = {0: "attribute_0", 1: "attribute_1", ...}.
    :param rf_mode: whether the random forest mode is active. 0 means no, 1 means that the computation will be performed
    using the random forest method. This is because in this case, we can select nfeat attributes
    each time we are going to make a split.
    :return: Tree created using the training set.
    """

    tree = Tree()
    tree.grow(x, y, nmin, minleaf, nfeat, attributes, rf_mode)
    return tree


def tree_pred(data: [[]], tree: Tree):
    """
    This function uses a single Tree to classify the data received in input.

    :param data: data to be classified.
    :param tree: tree used to classify the data in input.
    :return: classification of the data.
    """

    return tree.classify(data)


def tree_grow_b(x: [[]], y: [], nmin: int, minleaf: int, nfeat: int, m: int, attributes: dict, rf_mode: int):
    """
    This function creates a Forest object tailored for the given data.
    It can be used also to let grow a Bagging.
    A Forest is a List of Trees.

    :param x: array of data attributes and values passed without the classification labels.
    :param y: array of classifications in the same order of x.
    :param nmin: minimum number of observations (elements) in order to create a Node.
    :param minleaf: minimum number of observations (elements) in order to create a leaf.
    :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
    :param attributes: a dictionary that specifies the attributes available to make a split.
    Format: attributes = {0: "attribute_0", 1: "attribute_1", ...}.
    :param rf_mode: whether the random forest mode is active. 0 means no, 1 means that the computation will be performed
    using the random forest method. This is because in this case, we can select nfeat attributes
    each time we are going to make a split.
    :param m: number of trees to grow in the current forest.
    :return: Forest created using the training set.
    """

    forest = Forest()
    trees = forest.grow_b(x, y, nmin, minleaf, nfeat, m, attributes, rf_mode)
    return trees


def tree_pred_b(data: [[]], trees: [Tree]):
    """
    This function uses a Forest (or Bagging) to classify the data received in input.

    :param data: data to be classified.
    :param trees: list of Trees used to classify the data.
    :return: classification of the data.
    """
    forest = Forest()
    return forest.pred_b(data, trees)
