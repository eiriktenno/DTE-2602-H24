import csv
import random as rnd
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats  # Used for "mode" - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
from decision_tree_nodes import DecisionTreeBranchNode, DecisionTreeLeafNode
from matplotlib import lines
from numpy.typing import NDArray

# The code below is "starter code" for graded assignment 2 in DTE-2602
# You should implement every method / function which only contains "pass".
# "Helper functions" can be left unedited.
# Feel free to add additional classes / methods / functions to answer the assignment.
# You can use the modules imported above, and you can also import any module included
# in Python 3.10. See https://docs.python.org/3.10/py-modindex.html .
# Using any other modules / external libraries is NOT ALLOWED.


#########################################
#   Data input / prediction evaluation
#########################################


def read_from_csv(csv_file_path: str, delimiter: str) -> tuple:
    """Read the CSV file and returning its content.

    Args:
        csv_file_path:
            String, path for the csv file to read from.
        delimiter:
            String, how the csv file i seperated.

    
    Returns:
        tuple: A tuple containing two NumPy arrays:
        - First header. The first line read from the CSV file.
        - Second content. All other rows from the CSV file.
    
    """
    with open(csv_file_path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        header = next(csv_reader)
        content = []
        for x in csv_reader:
            content.append(x)
    return np.array(header), np.array(content)


def check_content_is_in_matrix(header: NDArray, desired_values: NDArray) -> list:
    """Checks if the desired values are present in the header and returns a list of True/False values.

    Args:
        header (NDArray): An numpy array containing the headers from a csv file.
        desired_values (NDArray): An numpy array containing a list of the desired values.

    Returns:
        check_list (list): List with True/False depending on presence in header and desired_values.
    """
    check_list = []
    for i in range(len(header)):
        if header[i] in desired_values:
            check_list.append(True)
        else:
            check_list.append(False)
    return check_list

def is_number(element) -> bool:
    """Check if the input element is a number or not.

    Args:
        element (NA): Variable to check.

    Returns:
        bool: True if it's a number, False if not.
    """
    try:
        float(element)
        return True
    except ValueError:
        return False

def remove_uncomplete_data(X: NDArray, header: NDArray, label_column: str) -> tuple[NDArray, NDArray]:
    """Removes uncomplete data (typically involving 'NA').

    Args:
        X (NDArray): The NumPy matrix with the content.
        header (NDArray): Headers for the data in X.
        label_column (str): The header for the label variable (y).

    Returns:
        tuple[NDArray, NDArray]: Returns a X and y, where all rows with uncomplete data has been
        removed.
    """
    y_check = np.array(check_content_is_in_matrix(header, np.array([label_column])))
    valid_rows = []

    y = X[:, y_check]
    X = X[:, ~y_check]

    for i in range(len(X)):
        valid_row = True
        for j in range(len(X[i])):
            if not is_number(X[i][j]):
                valid_row = False
        if valid_row:
            valid_rows.append(i)

    X_cleaned = X[valid_rows]
    y_cleaned = y[valid_rows]

    return X_cleaned, y_cleaned


def normalize_data(X: NDArray) -> NDArray:
    """Normalizing the data based on Z-score.
        Each feature will have a mean of 0, and a standard deviation of 1.

    Args:
        X (NDArray): Input matrix, to be normalized.

    Returns:
        NDArray: X matrix, with normalized data.
    """
    length = X.shape[1]
    X = np.array(X, dtype='f')
        
    for i in range(length):
        std = np.std(X[:, i])
        mean = np.mean(X[:, i])
        for z in range(len(X[:, i])):
            X[:, i][z] = ((X[:, i][z] - mean)/std)
    return X


def convert_text_to_index(y: NDArray) -> NDArray:
    """Converting unique text from y to unique numbers.

    Args:
        y (NDArray): NumPy Array, with text elements.

    Returns:
        NDArray: NumPy Array, with numbers instead of text.
    """
    unique_list = np.unique(y)
    unique_numeric = np.searchsorted(unique_list, y) # NB: ChatGPT har hjulpet til her.
    return unique_numeric


def read_data(path: str, delmiter: str, desired_columns: list[str], label_column: str) -> tuple[NDArray, NDArray]:
    """Read data from CSV file, remove rows with missing data, and normalize

    Parameters
    ----------
    path:
        String, path for the csv file to read from.
    delimiter:
        String, how the csv file i seperated.
    label:
        String, header name for the label-vector (y)

    Returns
    -------
    X: NDArray
        Numpy array, shape (n_samples,4), where n_samples is number of rows
        in the dataset. Contains the four numeric columns in the dataset
        (bill length, bill depth, flipper length, body mass).
        Each column (each feature) is normalized by subtracting the column mean
        and dividing by the column std.dev. ("z-score").
        Rows with missing data ("NA") are discarded.
    y: NDarray
        Numpy array, shape (n_samples,)
        Contains integer values (0, 1 or 2) representing the penguin species

    Notes
    -----
    Z-score normalization: https://en.wikipedia.org/wiki/Standard_score .
    """
    header, content = read_from_csv(path, delmiter)
    desired_columns = np.append(desired_columns, label_column)
    check_list = check_content_is_in_matrix(header, desired_columns)
    X, y = remove_uncomplete_data(content[:, check_list], header[check_list], label_column)
    X = normalize_data(X)
    y = convert_text_to_index(y)
    return X, y



def convert_y_to_binary(y: NDArray, y_value_true: int) -> NDArray:
    """Convert integer valued y to binary (0 or 1) valued vector

    Parameters
    ----------
    y: NDArray
        Integer valued NumPy vector, shape (n_samples,)
    y_value_true: int
        Value of y which will be converted to 1 in output.
        All other values are converted to 0.

    Returns
    -------
    y_binary: NDArray
        Binary vector, shape (n_samples,)
        1 for values in y that are equal to y_value_true, 0 otherwise
    """
    y_binary = []
    for i in range(len(y)):
        if y[i] == y_value_true:
            y_binary.append(1)
        else:
            y_binary.append(0)
    return np.array(y_binary)


######################## SLETTTTTTT
# def convert_y_to_int(y: NDArray, y_value_true: int) -> NDArray:
#     """Convert integer values

#     Parameters
#     ----------
#     y: NDArray
#         Integer valued NumPy vector, shape (n_samples,)

#     Returns
#     -------
#     y_int: NDArray
#         Integer vector, shape (n_samples,)
#         Converts each unique text to a unique number.
#     """
#     y_int = []
#     for i in range(len(y)):
#         if y[i] == y_value_true:
#             y_binary.append(1)
#         else:
#             y_binary.append(0)
#     return np.array(y_binary)
# ALLEREDE LAGET TIL.


def shuffel_data_set(X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Shuffel function. Is randomizing the order for X and y.

    Args:
        X (NDArray): X, the content matrix.
        y (NDArray): y, the label matrix.

    Returns:
        tuple[NDArray, np.NDArray]: Returns the shuffled arrays for X and y.
    """
    rng = np.random.default_rng()
    ind = rng.permutation(X.shape[0])
    X_shuffled = X[ind]
    y_shuffled = y[ind]

    return X_shuffled, y_shuffled.flatten()

def train_test_split(
    X: NDArray, y: NDArray, train_frac: float
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """Shuffle and split dataset into training and testing datasets

    Parameters
    ----------
    X: NDArray
        Dataset, shape (n_samples,n_features)
    y: NDArray
        Values to be predicted, shape (n_samples)
    train_frac: float
        Fraction of data to be used for training

    Returns
    -------
    (X_train,y_train): tuple[NDArray, NDArray]]
        Training dataset
    (X_test,y_test): tuple[NDArray, NDArray]]
        Test dataset
    """

    X, y = shuffel_data_set(X, y)

    split_index = int(len(X)*train_frac)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return (X_train, y_train), (X_test, y_test)


def accuracy(y_pred: NDArray, y_true: NDArray) -> float:
    """Calculate accuracy of model based on predicted and true values

    Parameters
    ----------
    y_pred: NDArray
        Numpy array with predicted values, shape (n_samples,)
    y_true: NDArray
        Numpy array with true values, shape (n_samples,)

    Returns
    -------
    accuracy: float
        Fraction of cases where the predicted values
        are equal to the true values. Number in range [0,1]

    # Notes:
    See https://en.wikipedia.org/wiki/Accuracy_and_precision#In_classification
    """
    correct_count = 0
    length = len(y_pred)
    for i in range(length):
        if y_pred[i] == y_true[i]:
            correct_count += 1
    accuracy = correct_count/(length)
    return accuracy


##############################
#   Gini impurity functions
##############################


def gini_impurity(y: NDArray) -> float:
    """Calculate Gini impurity of a vector

    Parameters
    ----------
    y: NDArray, integers
        1D NumPy array with class labels

    Returns
    -------
    impurity: float
        Gini impurity, scalar in range [0,1)

    # Notes:
    - Wikipedia ref.: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """

    n_samples = y.shape[0]
    unique_list, counts = np.unique(y, return_counts=True)
    
    sumarize = 0
    for i in range(len(unique_list)):
        sumarize += (counts[i]/n_samples)**2
    
    return 1 - sumarize



def gini_impurity_reduction(y: NDArray, left_mask: NDArray) -> float:
    """Calculate the reduction in mean impurity from a binary split

    Parameters
    ----------
    y: NDArray
        1D numpy array
    left_mask: NDArray
        1D numpy boolean array, True for "left" elements, False for "right"

    Returns
    -------
    impurity_reduction: float
        Reduction in mean Gini impurity, scalar in range [0,0.5]
        Reduction is measured as _difference_ between Gini impurity for
        the original (not split) dataset, and the _weighted mean impurity_
        for the two split datasets ("left" and "right").

    """
    gi_before_split = gini_impurity(y)
    left = y[left_mask]
    right= y[~left_mask]
    n_samples = y.shape[0]
    gi_left = gini_impurity(left)
    gi_right = gini_impurity(right)

    new_gi = (gi_left*(len(left)/n_samples)) + (gi_right*(len(right)/n_samples))

    impurity_reduction = gi_before_split - new_gi

    return float(impurity_reduction)




def best_split_feature_value(X: NDArray, y: NDArray) -> tuple[float, int, float]:
    """Find feature and value "split" that yields highest impurity reduction

    Parameters
    ----------
    X: NDArray
        NumPy feature matrix, shape (n_samples, n_features)
    y: NDArray
        NumPy class label vector, shape (n_samples,)

    Returns
    -------
    impurity_reduction: float
        Reduction in Gini impurity for best split.
        Zero if no split that reduces impurity exists.
    feature_index: int
        Index of X column with best feature for split.
        None if impurity_reduction = 0.
    feature_value: float
        Value of feature in X yielding best split of y
        Dataset is split using X[:,feature_index] <= feature_value
        None if impurity_reduction = 0.

    Notes
    -----
    The method checks every possible combination of feature and
    existing unique feature values in the dataset.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    question_feature_index = None
    question_value = None
    best_GI_reduction = (-np.inf)

    #for every feature (every column) f with index j in X:
    # for j in range(n_features):
    #     print(f"J: {j}")
    for j in range(n_features):
        f_j = X[:,j]

        unique_list, counts = np.unique(f_j, return_counts=True)
        for value in unique_list:
            left_mask = f_j <= value
            GI_reduction = gini_impurity_reduction(y, left_mask)
            if GI_reduction > best_GI_reduction:
                best_GI_reduction = GI_reduction
                question_feature_index = j
                question_value = value

    return best_GI_reduction, question_feature_index, question_value


###################
#   Perceptron
###################


class Perceptron:
    """Perceptron model for classifying two classes

    Attributes
    ----------
    weights: NDArray
        Array, shape (n_features,), with perceptron weights
    bias: float
        Perceptron bias value
    converged: bool | None
        Boolean indicating if Perceptron has converged during training.
        Set to None if Perceptron has not yet been trained.
    """

    
    def __init__(self, n_features: int, bias: float = 0):
        """Initialize perceptron"""
        self.weights = np.random.rand(n_features)
        self.bias = bias
        self.converged = None
        self.converged_index = 0


    def predict_single(self, X: NDArray) -> int:
        """Predict / calculate perceptron output for single observation / row x
        <Write rest of docstring here>
        Args:
            x (NDArray): Et datapunkt for hver feature som skal testes.
            F (int): Aktiveringsfunksjonene. Som standard "Threshold".

        Returns:
            int: _description_
        """
        I = 0
        for i in range(len(X)):
            I += X[i] * self.weights[i]
    
        I += self.bias
        
        return 1 if I >= 0 else 0

    def predict(self, X: NDArray) -> NDArray:
        """Predict / calculate perceptron output for data matrix X
        <Write rest of docstring here>
        """
        X_predict = []
        for i in range(X.shape[0]):
            if self.predict_single(X[i]) == 0:
                X_predict.append(0)
            else:
                X_predict.append(1)
        return np.array(X_predict)

    def train(self, X: NDArray, y: NDArray, learning_rate: float = 0.01, max_epochs: int = 1000):
        """Train perceptron on dataset X with binary labels y

        Args:
            X (NDArray): Training data with shape (n_samples, n_features)
            y (NDArray): Labels with shape (n_samples,)
            learning_rate (float): The learning rate (default=0.01)
            max_epochs (int): Maximum number of epochs (default=10000)
        """
        self.converged = False
        
        for epoch in range(max_epochs):
            errors = 0
            for i in range(len(X)):
                V = self.predict_single(X[i])

                self.weights += learning_rate * (y[i] - V) * X[i]
                self.bias += learning_rate * (y[i] - V) * 1

                if y[i] != V:
                    errors += 1

            if errors == 0:
                self.converged = True
                self.converged_index = epoch
                break
            #print(f"Epoch {epoch}, Weights: {self.weights}, Bias: {self.bias}")


    def decision_boundary_slope_intercept(self) -> tuple[float, float]:
        """Calculate slope and intercept for decision boundary line (2-feature data only)
        <Write rest of docstring here>
        """
        # slope = -(weights[0]/weights[2])/(weights[0]/weights[1])  
        # intercept = -weights[0]/weights[2]

        slope = -(self.bias/self.weights[1])/(self.bias/self.weights[0])  
        intercept = -self.bias/self.weights[1]
        return slope, intercept


####################
#   Decision tree
####################


class DecisionTree:
    """Decision tree model for classification

    Attributes
    ----------
    _root: DecisionTreeBranchNode | None
        Root node in decision tree
    """

    def __init__(self):
        """Initialize decision tree"""
        self._root = None

    def __str__(self) -> str:
        """Return string representation of decision tree (based on binarytree.Node.__str__())"""
        if self._root is not None:
            return str(self._root)
        else:
            return "<Empty decision tree>"

    def fit(self, X: NDArray, y: NDArray):
        """Train decision tree based on labelled dataset

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray, integers
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        Creates the decision tree by calling _build_tree() and setting
        the root node to the "top" DecisionTreeBranchNode.

        """
        self._root = self._build_tree(X, y)

    def _build_tree(self, X: NDArray, y: NDArray):
        """Recursively build decision tree

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        - Determines the best possible binary split of the dataset. If no impurity
        reduction can be achieved, a leaf node is created, and its value is set to
        the most common class in y. If a split can achieve impurity reduction,
        a decision (branch) node is created, with left and right subtrees created by
        recursively calling _build_tree on the left and right subsets.

        """
        # Find best binary split of dataset
        impurity_reduction, feature_index, feature_value = best_split_feature_value(
            X, y
        )

        # If impurity can't be reduced further, create and return leaf node
        if impurity_reduction == 0:
            leaf_value = scipy.stats.mode(y, keepdims=False)[0]
            return DecisionTreeLeafNode(leaf_value)

        # If impurity _can_ be reduced, split dataset, build left and right
        # branches, and return branch node.
        else:
            left_mask = X[:, feature_index] <= feature_value
            left = self._build_tree(X[left_mask], y[left_mask])
            right = self._build_tree(X[~left_mask], y[~left_mask])
            return DecisionTreeBranchNode(feature_index, feature_value, left, right)

    def predict(self, X: NDArray):
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray, integers
            NumPy class label vector (predicted), shape (n_samples,)
        """
        if self._root is not None:
            return self._predict(X, self._root)
        else:
            raise ValueError("Decision tree root is None (not set)")

    def _predict(
        self, X: NDArray, node: Union["DecisionTreeBranchNode", "DecisionTreeLeafNode"]
    ) -> NDArray:
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        node: "DecisionTreeBranchNode" or "DecisionTreeLeafNode"
            Node used to process the data. If the node is a leaf node,
            the data is classified with the value of the leaf node.
            If the node is a branch node, the data is split into left
            and right subsets, and classified by recursively calling
            _predict() on the left and right subsets.

        Returns
        -------
        y: NDArray
            NumPy class label vector (predicted), shape (n_samples,)

        Notes
        -----
        The prediction follows the following logic:

            if the node is a leaf node
                return y vector with all values equal to leaf node value
            else (the node is a branch node)
                split the dataset into left and right parts using node question
                predict classes for left and right datasets (using left and right branches)
                "stitch" predictions for left and right datasets into single y vector
                return y vector (length matching number of rows in X)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        if isinstance(node, DecisionTreeBranchNode): # NB: Fikk hjelp fra chatgpt for løsning for å forstå Union/isinstance.
            # Is splitting X first based on the question, we wan to ask (feature_index)
            # Then splitting to left/right based on the value
            left_mask = X[:,node.feature_index]<=node.feature_value
            right_mask = X[:,node.feature_index]>node.feature_value

            left_branch = X[left_mask]
            right_branch = X[right_mask]

            # Recursive prediction for each of the new sets.
            left_predict = self._predict(left_branch, node.left)
            right_predict = self._predict(right_branch, node.right)

            y = np.zeros(n_samples)
            y[left_mask] = left_predict
            y[right_mask] = right_predict

            return y

        elif isinstance(node, DecisionTreeLeafNode):
            y = []
            for i in range(X.shape[0]):
                y.append(node.value)
            return np.array(y)
        else:
            raise ValueError("Decision tree root is None (not set)")
        


############
#   MAIN
############

def plot_experiments(
        title: str, 
        p: Perceptron, 
        X_test: NDArray, 
        y_test: NDArray, 
        desired_columns: NDArray
        ):
    """_summary_

    Args:
        p (Perceptron): _description_
        X_test (NDArray): _description_
        desired_columns (NDArray): _description_
    """
    plt.xlabel(desired_columns[0])
    plt.ylabel(desired_columns[1])
    plt.title(title)

    for label_value in np.unique(y_test):
        plt.scatter(x=X_test[y_test==label_value, 0],
                    y=X_test[y_test==label_value, 1])

    x_min, x_max = np.min(X_test[:, 0]), np.max(X_test[:, 0])
    x_points = np.linspace(x_min, x_max, 100)
    y_points = -(p.weights[0] * x_points + p.bias) / p.weights[1]

    plt.plot(x_points, y_points, 'k-', label='Decision boundary')
    plt.show()

def experiment_1():
    print("EXPERIMENT 1")
    p = Perceptron(2, 1)
    desired_columns = np.array(['bill_depth_mm', 'flipper_length_mm'])
    label_column = 'species'
    X, y = read_data("palmer_penguins.csv", ",", desired_columns, label_column)
    y_binary = convert_y_to_binary(y, 2)

    train, test = train_test_split(X, y_binary, 0.8)
    X_train = train[0]
    X_test = test[0]

    y_train = train[1]
    y_test = test[1]

    p.train(X_train, y_train)

    plot_experiments('Experiment 1 - TRAIN',p, X_train, y_train, desired_columns)
    plot_experiments('Experiment 1 - TEST',p, X_test, y_test, desired_columns)

    predicted_list = p.predict(X_test)

    print(f"ACCURACY: {accuracy(predicted_list, y_test)}, CONVERGED: {p.converged} Index: {p.converged_index}")


def experiment_2():
    print("EXPERIMENT 2")
    p = Perceptron(2, 1)
    desired_columns = np.array(['bill_length_mm', 'flipper_length_mm'])
    label_column = 'species'
    X, y = read_data("palmer_penguins.csv", ",", desired_columns, label_column)
    y_binary = convert_y_to_binary(y, 1)

    train, test = train_test_split(X, y_binary, 0.8)
    X_train = train[0]
    X_test = test[0]

    y_train = train[1]
    y_test = test[1]

    p.train(X_train, y_train)

    plot_experiments('Experiment 2 - TRAIN',p, X_train, y_train, desired_columns)
    plot_experiments('Experiment 2 - TEST',p, X_test, y_test, desired_columns)

    predicted_list = p.predict(X_test)

    print(f"ACCURACY: {accuracy(predicted_list, y_test)}, CONVERGED: {p.converged} Index: {p.converged_index}")


def experiment_3():
    print("EXPERIMENT 3")
    #desired_columns = np.array(['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    desired_columns = np.array(['bill_depth_mm', 'flipper_length_mm'])
    label_column = 'species'
    X, y = read_data("palmer_penguins.csv", ",", desired_columns, label_column)
    y_binary = convert_y_to_binary(y, 2)

    train, test = train_test_split(X, y_binary, 0.8)
    X_train = train[0]
    X_test = test[0]
    y_train = train[1]
    y_test = test[1]

    dt_classifier = DecisionTree()
    dt_classifier.fit(X_train, y_train)

    predicted_list = dt_classifier.predict(X_test)

    print(dt_classifier)
    print(f"ACCURACY: {accuracy(predicted_list, y_test)}")


def experiment_4():
    print("EXPERIMENT 4")
    #desired_columns = np.array(['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    desired_columns = np.array(['bill_length_mm', 'bill_depth_mm'])
    label_column = 'species'
    X, y = read_data("palmer_penguins.csv", ",", desired_columns, label_column)
    y_binary = convert_y_to_binary(y, 1)

    train, test = train_test_split(X, y_binary, 0.8)
    X_train = train[0]
    X_test = test[0]
    y_train = train[1]
    y_test = test[1]

    dt_classifier = DecisionTree()
    dt_classifier.fit(X_train, y_train)

    predicted_list = dt_classifier.predict(X_test)

    print(dt_classifier)
    print(f"ACCURACY: {accuracy(predicted_list, y_test)}")

def experiment_5():
    print("EXPERIMENT 5")
    n_tests = int(input("How many tests?\n"))

    accuracy_list = []

    for i in range(n_tests):
        desired_columns = np.array(['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
        label_column = 'species'
        X, y = read_data("palmer_penguins.csv", ",", desired_columns, label_column)

        train, test = train_test_split(X, y, 0.8)
        X_train = train[0]
        X_test = test[0]
        y_train = train[1]
        y_test = test[1]

        dt_classifier = DecisionTree()
        dt_classifier.fit(X_train, y_train)

        predicted_list = dt_classifier.predict(X_test)

        accuracy_list.append(accuracy(predicted_list, y_test))

    average = sum(accuracy_list) / len(accuracy_list)
    print(f"The average is: {average}")

if __name__ == "__main__":
    # Demonstrate your code / solutions here.
    # Be tidy; don't cut-and-paste lots of lines.
    # Experiments can be implemented as separate functions that are called here.

    print(
        """
        Experiments that can be simulated:

        1. Perceptron: Separate Gentoo from the other two species. 
           Features to be used: bill_depth_mm and flipper_length_mm. Show plot and accuracy.

        2. Perceptron: Separate the Chinstrap species from the other two. 
           Features to be used: bill_length_mm and bill_depth_mm. Show plot and accuracy.

        3. Create a decision tree to separate the Gentoo penguin species from the other two. 
           Features to be used: bill_depth_mm and flipper_length_mm. Measure accuracy and visualize the decision tree.

        4. Create a decision tree to separate the Chinstrap species from the others. 
           Features to be used: bill_length_mm and bill_depth_mm. Measure accuracy and visualize the decision tree.

        5. Repeat the experiments several times with random shuffling and splitting. 
           Each time, create a decision tree based on all 4 features in the dataset to distinguish between all three species. 
           Measure accuracy.
        """ 
    )

    choise = int(input("Which Experiment to simulate?\n"))

    print(choise)

    if choise == 1:
        experiment_1()
    elif choise == 2:
        experiment_2()
    elif choise == 3:
        experiment_3()
    elif choise == 4:
        experiment_4()
    elif choise == 5:
        experiment_5()
