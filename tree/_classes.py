from ._utils import _check_param, get_numpy_array
from sklearn.base import BaseEstimator
from ._tree import DecisionTreeRegressorSlow

class TreeRegressorSlow(BaseEstimator, DecisionTreeRegressorSlow):
    def __init__(self, max_depth = 10, min_samples_leaf = 1, min_samples_split = 1):
        # checking input parameters
        _check_param(max_depth, min_samples_leaf, min_samples_split)

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tree = {}

    def fit(self, X_train, y_train):
        # checking input data
        X_train, y_train = get_numpy_array(X_train, y_train)
        X_train, y_train = super(DecisionTreeRegressorSlow, self)._check_input(X_train, y_train)

        # fit tree
        self.tree = super(TreeRegressorSlow, self)._build(X_train, y_train, self.tree, self.max_depth)
        return self

    def predict(self, X_test):
        X_test = super(DecisionTreeRegressorSlow, self)._check_input_test(X_test)
        return super(TreeRegressorSlow, self)._predict(X_test, self.tree)
