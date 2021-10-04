from Assignment_1.Utils import best_split_all_attributes
import numpy as np
import random


class Node:
    """
    Class created to define a Node object, it belongs to a Tree and it can be: root, leaf or an intermediate Node.
    When the Tree is growing, new nodes are generated and recursively classified.
    """

    def __init__(self, root: bool = False, leaf: bool = False, classification: int = None):
        """
        Constructor for the class Node: root Node means that the Node is the beginning of the tree.
        A Node can have a right Node and a left Node (children), if it is not a leaf.
        If the Node is a leaf it has the value of the classification (1, 0): if the value is None,
        then it is not a leaf and it can be split, hence, it has the attribute used to divide the data,
        which is an int encoded as: 0: attribute_0, 1: attribute_1, ...
        and the value of the split (e.g. split the data based on a numerical attribute: < 36.5, then, 36.5 is the
        value of the split).

        :param root: True if node is root, False otherwise.
        :param leaf: True if node is leaf (pure), False otherwise.
        :param classification: if leaf == True, here there is the value of classification (binary).
        """

        self.is_root = root
        self.is_leaf = leaf
        self.classification = classification
        self.value = None
        self.attribute = None
        self.left = None
        self.right = None


    def split(self, data_x: [[]], data_y: [], nmin: int, minleaf: int, nfeat: int, attributes: dict, rf_mode: int):
        """
        This function made the Tree grows by splitting the data in input into 2 different groups.
        To do this, it iterates through nfeat attributes and calculate the best split of each.
        Then, the attribute that gives an impurity reduction greater than the others,
        would be selected as attribute to split the data, and it is saved together with its split value.
        Then, if one of the child nodes classifies as a leaf, it generates a leaf Node.
        Otherwise, it generates a normal Node.

        :param data_x: array of data attributes and values passed without the classification labels.
        :param data_y: array of classifications in the same order of data_x.
        :param nmin: minimum number of observations (elements) in order to create a Node.
        :param minleaf: minimum number of observations (elements) in order to create a leaf.
        :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
        :param attributes: a dictionary that specifies the attributes available to make a split.
        Format: attributes = {0: "attribute_0", 1: "attribute_1", ...}.
        :param rf_mode: whether the random forest mode is active. 0 means no, 1 means that the computation will be performed
        using the random forest method. This is because in this case, we can select nfeat attributes
        each time we are going to make a split.
        :return: None, it simply stores in its variables the created nodes and their connections.
        """

        if rf_mode:
            random_forest_attributes = random.sample(list(attributes), nfeat)
            best_attribute, best_split, best_impurity_left, best_impurity_right = best_split_all_attributes(data_x, data_y, minleaf, random_forest_attributes)

            # if maybe we had bad luck with the first attributes selection, try again with 6 different attributes:
            while best_attribute is None and best_split is None:
                random_forest_attributes = random.sample(list(attributes), nfeat)
                best_attribute, best_split, best_impurity_left, best_impurity_right = best_split_all_attributes(data_x, data_y, minleaf, random_forest_attributes)
        else:
            best_attribute, best_split, best_impurity_left, best_impurity_right = best_split_all_attributes(data_x, data_y, minleaf, attributes)


        self.attribute = best_attribute
        self.value = best_split


        # if no split is allowed on the given attributes, then the node is a leaf
        if self.attribute is None and self.value is None:
            unique, counts = np.unique(data_y, return_counts=True)
            count_0_and_1 = dict(zip(unique, counts))

            # if nothing was classified as 0:
            if 0 not in count_0_and_1:
                count_0_and_1[0] = 0

            # if nothing was classified as 1:
            if 1 not in count_0_and_1:
                count_0_and_1[1] = 0

            # if the number of 1s is equal to number of 0s: classify as 0
            # else, classify with the major class.
            if count_0_and_1[0] == count_0_and_1[1]:
                classification = 0
            else:
                classification = max(count_0_and_1, key=count_0_and_1.get)

            self.is_leaf = True
            self.classification = classification

        else:
            # if we have a split and attribute value,
            # sort the data left and right based on the best split
            left_node_data = data_x[:, best_attribute] <= best_split
            right_node_data = data_x[:, best_attribute] > best_split

            # checking the left child

            if best_impurity_left == 0 or len(data_x[left_node_data]) < nmin:

                unique, counts = np.unique(data_y[left_node_data], return_counts=True)
                count_0_and_1 = dict(zip(unique, counts))

                # if nothing was classified as 0:
                if 0 not in count_0_and_1:
                    count_0_and_1[0] = 0

                # if nothing was classified as 1:
                if 1 not in count_0_and_1:
                    count_0_and_1[1] = 0

                # if the number of 1s is equal to number of 0s: classify as 0
                # else, classify with the major class.
                if count_0_and_1[0] == count_0_and_1[1]:
                    classification = 0
                else:
                    classification = max(count_0_and_1, key=count_0_and_1.get)

                self.left = Node(leaf=True, classification=classification)

            else:
                self.left = Node()
                self.left.split(data_x[left_node_data], data_y[left_node_data], nmin, minleaf, nfeat, attributes, rf_mode)

            # checking the right child

            if best_impurity_right == 0 or (len(data_x[right_node_data]) < nmin):

                unique, counts = np.unique(data_y[right_node_data], return_counts=True)
                count_0_and_1 = dict(zip(unique, counts))

                # if nothing was classified as 0:
                if 0 not in count_0_and_1:
                    count_0_and_1[0] = 0

                # if nothing was classified as 1:
                if 1 not in count_0_and_1:
                    count_0_and_1[1] = 0

                # if the number of 1s is equal to number of 0s: classify as 0
                # else, classify with the major class.
                if count_0_and_1[0] == count_0_and_1[1]:
                    classification = 0
                else:
                    classification = max(count_0_and_1, key=count_0_and_1.get)

                self.right = Node(leaf=True, classification=classification)

            else:
                self.right = Node()
                self.right.split(data_x[right_node_data], data_y[right_node_data], nmin, minleaf, nfeat, attributes, rf_mode)


    def classify(self, data: []):
        """
        Recursive method that finds the correct leaf for new data, it sends the data
        to the predicted child node returning the classification value for those data.

        :param data: array of data attributes and values to be classified, passed without the classification labels.
        :return: classification labels of the data array received in input.
        """

        if self.is_leaf:
            return self.classification
        else:
            if data[self.attribute] < self.value:
                return self.left.classify(data)
            else:
                return self.right.classify(data)


class Tree:
    """
    Class created to define a Tree object, and it simply has a pointer to a Node,
    which is the root of the Tree.
    From the root, as in the classic binary Tree data-structure, all the stored data are accessible
    through pointers to the left or right child nodes.
    """

    def __init__(self):
        """
        Constructor for the class Tree that generates a root Node object.
        """

        self.root = Node(root=True)


    def grow(self, data_x: [[]], data_y: [], nmin: int, minleaf: int, nfeat: int, attributes: dict, rf_mode: int):
        """
        This function grows the tree using the x_train data (here: data_x) together with their labels y_train (here: data_y).
        How the tree grows: recursively, it calls the method split from the class Node.

        :param data_x: array of data attributes and values passed without the classification labels.
        :param data_y: array of classifications in the same order of x.
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

        self.root.split(data_x, data_y, nmin=nmin, minleaf=minleaf, nfeat=nfeat, attributes=attributes, rf_mode=rf_mode)


    def classify(self, data_x: [[]]):
        """
        This function uses a single Tree to classify the data received in input.

        :param data: data to be classified.
        :param tree: tree used to classify the data in input.
        :return: classification of the data.
        """

        predictions = []

        for observation in data_x:
            predictions.append(self.root.classify(observation))

        return predictions


class Forest:
    """
    Class created to define a Forest object. A Forest is defined as a List of Trees.
    """

    def __init__(self):
        """
        Constructor for the class Forest.
        """
        self.forest = []
        self.classification = []


    def grow_b(self, x: [[]], y: [], nmin: int, minleaf: int, nfeat: int, m: int, attributes: dict, rf_mode: int):
        """
        This function let grows a Forest.
        It can be used also to let grow a Bagging.

        :param x: array of data attributes and values passed without the classification labels.
        :param y: array of classifications in the same order of x.
        :param nmin: minimum number of observations (elements) in order to create a Node.
        :param minleaf: minimum number of observations (elements) in order to create a leaf.
        :param nfeat: number of attributes that are used to get the best attribute for splitting the data.
        :param m: number of trees to grow in the current forest.
        :param attributes: a dictionary that specifies the attributes available to make a split.
        Format: attributes = {0: "attribute_0", 1: "attribute_1", ...}.
        :param rf_mode: whether the random forest mode is active. 0 means no, 1 means that the computation will be performed
        using the random forest method. This is because in this case, we can select nfeat attributes
        each time we are going to make a split.
        :return: Forest created using the training set.
        """

        # averaging with bootstrapping: randomly select the used data
        for i in range(m):
            new_x_data, new_y_data = np.empty_like(x), np.empty_like(y)
            for j in range(len(y)):
                observation = random.randint(0, len(y)-1)
                new_x_data[j] = x[observation]
                new_y_data[j] = y[observation]

            tree = Tree()
            tree.grow(new_x_data, new_y_data, nmin, minleaf, nfeat, attributes, rf_mode)
            self.forest.append(tree)

            if i % 5 == 0 and i != 0:
                print("Trees evaluated by far:", i, "/", m)
            if i == m - 1:
                print("Trees evaluated by far:", i + 1, "/", m)

        return self.forest


    def pred_b(self, data, trees):
        """
        This function uses a Forest (or Bagging) to classify the data received in input.

        :param data: data to be classified.
        :param trees: list of Trees used to classify the data.
        :return: classification of the data.
        """

        self.classification = np.zeros(len(data))
        count = []

        for tree in trees:
            count.append(tree.classify(data))
        count = np.array(count)

        for column in range(len(count[0])):

            unique, counts = np.unique(count[:, column], return_counts=True)
            count_0_and_1 = dict(zip(unique, counts))

            # if nothing was classified as 0:
            if 0 not in count_0_and_1:
                count_0_and_1[0] = 0

            # if nothing was classified as 1:
            if 1 not in count_0_and_1:
                count_0_and_1[1] = 0

            # if the number of 1s is equal to number of 0s: classify as 0
            # else, classify with the major class.
            if count_0_and_1[0] == count_0_and_1[1]:
                classification = 0
            else:
                classification = max(count_0_and_1, key=count_0_and_1.get)

            self.classification[column] = classification

        return self.classification
