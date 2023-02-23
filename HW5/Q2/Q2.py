# I used these sites:
# https://www.geeksforgeeks.org/decision-tree-implementation-python/
# https://scikit-learn.org/stable/modules/tree.html
# https://www.datacamp.com/tutorial/decision-tree-classification-python
# https://www.w3schools.com/python/python_ml_decision_tree.asp
# https://towardsdatascience.com/decision-tree-algorithm-in-python-from-scratch-8c43f0e40173
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
import math

Name = "Ali Nazari"
Student_Number = "99102401"


def deep_copy_and_concat(x, y, level):
    first_value = copy.deepcopy(x)
    second_value = copy.deepcopy(y)
    return np.concatenate((first_value, second_value), axis=level)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, info=None):
        """
        Class for storing Decision Tree as a binary-tree
        Inputs:
        - feature: Name of the the feature based on which this node is split
        - threshold: The threshold used for splitting this subtree
        - left: left Child of this node
        - right: Right child of this node
        - value: Predicted value for this node (if it is a leaf node)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.info = info

    @staticmethod
    def is_leaf():
        return True


class Helper:
    @staticmethod
    def check_equality(main_obj, feature):
        if feature == "value":
            if main_obj.value is not None:
                return True
        return False

    @staticmethod
    def choose_side(independent_var, main_obj):
        value = independent_var[main_obj.feature]
        if value <= main_obj.threshold:
            return "left"
        return "right"

    @staticmethod
    def construct_tree(data, feature, threshold, side):
        to_return = list()
        if side == "left":
            for i in data:
                if i[feature] <= threshold:
                    to_return.append(i)
        else:
            for i in data:
                if i[feature] > threshold:
                    to_return.append(i)
        return np.array(to_return)

    @staticmethod
    def delete_repeated(a_list):
        to_return = list()
        for i in a_list:
            if i not in to_return:
                to_return.append(i)
        return np.asarray(to_return)

    @staticmethod
    def calculate_entropy(i, y):
        return len(y[y == i]) / len(y)

    @staticmethod
    def get_weight(first, total):
        return len(first) / len(total)

    @staticmethod
    def information_using_entropy(data, total, side):
        if side == "total":
            return DecisionTree.entropy(total)
        elif side == "left":
            return Helper.get_weight(data, total) * DecisionTree.entropy(data)
        else:
            return Helper.get_weight(data, total) * DecisionTree.entropy(data)

    @staticmethod
    def create_obj(feature, threshold, left, right, value, info):
        return Node(feature, threshold, left, right, value, info)

    @staticmethod
    def get_slice(data, i):
        return data[:, i]

    @staticmethod
    def thresholds(data, i):
        a_slice = Helper.get_slice(data, i)
        not_repeated = Helper.delete_repeated(a_slice)
        return not_repeated

    @staticmethod
    def check_not_empty(data):
        return len(data) > 0

    @staticmethod
    def temp_needed(this_max, max_one, i, j, left_one, right_one, to_return):
        if this_max > max_one:
            return Helper.create_obj(i, j, left_one, right_one, None, this_max), this_max
        return to_return, max_one

    @staticmethod
    def init_tree(x, y, data):
        if x is None and y is None:
            return data[:, -1], np.shape(data[:, :-1])[0], np.shape(data[:, :-1])[1]
        else:
            return y, np.shape(x)[0], np.shape(x)[1]


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Class for implementing Decision Tree
        Attributes:
        - max_depth: int
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until
            all leaves contain less than min_samples_split samples.
        - min_num_samples: int
            The minimum number of samples required to split an internal node
        - root: Node
            Root node of the tree; set after calling fit.
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def is_predicting_finished(self, independent_var, main_obj):
        """
        Criteria for continuing or finishing splitting a node
        Inputs:
        - depth: depth of the tree so far
        - num_class_labels: number of unique class labels in the node
        - num_samples: number of samples in the node
        :return: bool
        """
        if Helper.check_equality(main_obj, "value"):
            return main_obj.value
        result = Helper.choose_side(independent_var, main_obj)
        if result == "left":
            return self.is_predicting_finished(independent_var, main_obj.left)
        return self.is_predicting_finished(independent_var, main_obj.right)

    @staticmethod
    def split(feature, threshold, data):
        """
        Splitting X and y based on value of feature with respect to threshold;
        i.e., if x_i[feature] <= threshold, x_i and y_i belong to X_left and y_left.
        Inputs:
        - X: Array of shape (N, D) (number of samples and number of features respectively), samples
        - y: Array of shape (N,), labels
        - feature: Name of the the feature based on which split is done
        - threshold: Threshold of splitting
        :return: X_left, X_right, y_left, y_right
        """
        left_one = Helper.construct_tree(data, feature, threshold, "left")
        right_one = Helper.construct_tree(data, feature, threshold, "right")
        if Helper.check_not_empty(left_one) and Helper.check_not_empty(right_one):
            return left_one, right_one, data[:, -1], left_one[:, -1], right_one[:, -1]
        return left_one, right_one, None, None, None

    @staticmethod
    def entropy(y):
        """
        Computing entropy of input vector
        - y: Array of shape (N,), labels
        :return: entropy of y
        """
        not_repeated_items = Helper.delete_repeated(y)
        to_return = 0
        for i in not_repeated_items:
            term = Helper.calculate_entropy(i, y)
            to_return += -1 * term * math.log2(term)
        return to_return

    @staticmethod
    def information_gain(x, y, feature):
        """
        Returns information gain of splitting data with feature and threshold.
        Hint! use entropy of y, y_left and y_right.
        """
        to_return = Helper.information_using_entropy(None, feature, "total") - \
            Helper.information_using_entropy(x, feature, "left") - \
            Helper.information_using_entropy(y, feature, "right")
        return to_return

    @staticmethod
    def best_split(y, data):
        """
        Used for finding best feature and best threshold for splitting
        Inputs:
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        :return:
        """
        to_return = None
        max_one = -1 * math.inf
        for i in range(y):
            not_repeated = Helper.thresholds(data, i)
            for j in not_repeated:
                left_one, right_one, y, left_y, right_y = DecisionTree.split(i, j, data)
                if y is not None and left_one is not None and right_one is not None:
                    this_max = DecisionTree.information_gain(left_y, right_y, y)
                    to_return, max_one = Helper.temp_needed(this_max, max_one, i, j, left_one, right_one, to_return)
        return to_return

    def build_tree(self, x, y, data, depth=0):
        """
        Recursive function for building Decision Tree.
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        - depth: depth of tree so far
        :return: root node of subtree
        """
        y, number_of_samples, number_of_features = Helper.init_tree(x, y, data)
        if number_of_samples >= self.min_samples_split and depth <= self.max_depth:
            split_data = DecisionTree.best_split(number_of_features, data)
            if split_data.info > 0:
                left_subtree = self.build_tree(None, None, split_data.left, depth + 1)
                right_subtree = self.build_tree(None, None, split_data.right, depth + 1)
                return Helper.create_obj(split_data.feature, split_data.threshold, left_subtree, right_subtree, None,
                                         split_data.info)
        return Helper.create_obj(None, None, None, None, max(list(y), key=(list(y)).count), None)

    def fit(self, x, y):
        """
        Builds Decision Tree and sets root node
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        """
        self.root = self.build_tree(x, y, deep_copy_and_concat(x, y, 1))

    def predict(self, x):
        """
        Returns predicted labels for samples in X.
        :param x: Array of shape (N, D), samples
        :return: predicted labels
        """
        to_return = list()
        for i in x:
            to_return.append(self.is_predicting_finished(i, self.root))
        return to_return


main_data = pd.read_csv("breast_cancer.csv")
independent_variable = main_data.iloc[:, :-1].values
dependent_variable = main_data.iloc[:, -1].values.reshape(len(independent_variable), 1)
x_train, x_test, y_train, y_test = train_test_split(independent_variable, dependent_variable, test_size=.25,
                                                    random_state=1)
tree = DecisionTree(3)
tree.fit(independent_variable, dependent_variable)
test_data = pd.read_csv("test.csv")
prediction = tree.predict(test_data.values)
df = pd.DataFrame({'target': prediction})
df.to_csv("output.csv", index=False)
