from Utils import compute_best_split


class Node:
    """Class created to define a Node object, it belongs to a Tree and it can be: root, leaf or an intermediate Node.
    When the Tree is growing, new nodes are generated and recursively classified.
    """

    def __init__(self, root: bool = False, leaf: bool = False, classification: int = None):
        """
        Constructor for the class Node: root Node means that the Node is the beginning of the tree.
        A Node can have a right Node and a left Node (childs), if it is not a leaf.
        If the Node is a leaf it has the value of the classification (1, 0): if the value is None,
        then it is not a leaf and it can be split, hence, it has the attribute used to divide the data
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


    # make a recursive function to classify the data



    # make a function to split the data
    def split(self, data_x: [[]], data_y: [], nmin: int, minleaf: int, nfeat: int = None):
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
        :return: None, it simply stores in its variables the created nodes and their connections.
        """

        # use the compute_best_split function as it has been already implemented




        None






# create a Tree class object ---Daniele

class Tree:
    """Class created to define a Tree object, and it simply has a pointer to a Node,
    which is the root of the Tree.
    From the root, as in the classic binary Tree data-structure, all the stored data are accessible
    through pointers to the left or right child nodes.
    """

    def __init__(self):
        """Constructor for the class Tree that generates a root Node object."""
        self.root = Node(root=True)
