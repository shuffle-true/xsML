from ._utils import _check_param
from ._utils import _check_param_adaptive
from ._utils import get_numpy_array_train
from ._utils import get_numpy_array_test
from ._utils import convert_type_train
from ._utils import convert_type_test

from sklearn.base import BaseEstimator
from ._tree import DecisionTreeRegressorSlow, DecisionTreeRegressorFast, DecisionTreeAdaptive


class TreeRegressorSlow(BaseEstimator, DecisionTreeRegressorSlow):
    def __init__(self, max_depth=10, min_samples_leaf=1, min_samples_split=1):
        # checking input parameters
        _check_param(max_depth, min_samples_leaf, min_samples_split)

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tree = {}

    def fit(self, X_train, y_train):
        # checking input data

        # get numpy array from different type
        X_train, y_train = get_numpy_array_train(X_train, y_train)

        # get float64 type
        X_train, y_train = convert_type_train(X_train, y_train)

        # y_train must have 1 dimension
        if len(y_train.shape) != 1:
            raise IndexError(f"Argument 'y_train' not 1 dimension. 'y_train' dim {len(y_train.shape)} != dim 1. "
                             f"Please, check input data")

        # check lenght data
        if X_train.shape[0] != len(y_train):
            raise IndexError(f"Argument length 'X_train' != 'y_train': {X_train.shape[0]} != {len(y_train)}. "
                             f"Please, check input data.")

        X_train, y_train = super(DecisionTreeRegressorSlow, self)._check_input(X_train, y_train)

        # fit tree
        self.tree = super(TreeRegressorSlow, self)._build(X_train, y_train, self.tree, self.max_depth)
        return self

    def predict(self, X_test):
        X_test = get_numpy_array_test(X_test)

        X_test = convert_type_test(X_test)

        X_test = super(DecisionTreeRegressorSlow, self)._check_input_test(X_test)
        return super(TreeRegressorSlow, self)._predict(X_test, self.tree)


class TreeRegressor(BaseEstimator, DecisionTreeRegressorFast):
    def __init__(self, max_depth = 10, min_samples_leaf = 1, min_samples_split = 1):
        # checking input parameters
        _check_param(max_depth, min_samples_leaf, min_samples_split)

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tree = {}

    def fit(self, X_train, y_train):
        # checking input data

        # get numpy array from different type
        X_train, y_train = get_numpy_array_train(X_train, y_train)

        # get float64 type
        X_train, y_train = convert_type_train(X_train, y_train)

        # y_train must have 1 dimension
        if len(y_train.shape) != 1:
            raise IndexError(f"Argument length 'X_train' != 'y_train': {X_train.shape[0]} != {len(y_train)}. "
                             f"Please, check input data.")

        # check lenght data
        if X_train.shape[0] != len(y_train):
            raise IndexError(f"Argument length 'X_train' != 'y_train': {X_train.shape[0]} != {len(y_train)}. "
                             f"Please, check input data.")

        X_train, y_train = super(DecisionTreeRegressorFast, self)._check_input(X_train, y_train)

        # fit tree
        self.tree = super(TreeRegressor, self)._build(X_train, y_train, self.tree, self.max_depth)
        return self

    def predict(self, X_test):
        X_test = get_numpy_array_test(X_test)

        X_test = convert_type_test(X_test)

        X_test = super(DecisionTreeRegressorFast, self)._check_input_test(X_test)

        return super(TreeRegressor, self)._predict(X_test, self.tree)

class TreeRegressorAdaptive(BaseEstimator, DecisionTreeAdaptive):
    def __init__(self, max_depth = 10,
                 min_samples_leaf = 1,
                 min_samples_split = 1,
                 adaptive = True,
                 n_combinations = 2,
                 randomization = 'sum'):

        # checking input parameters
        _check_param(max_depth, min_samples_leaf, min_samples_split)
        _check_param_adaptive(adaptive, n_combinations, randomization)

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.adaptive = adaptive
        if self.adaptive:
            self.adaptive_test = True
        else:
            self.adaptive_test = False
        self.n_combinations = n_combinations,
        self.randomization = randomization,
        self.tree = {}

    def fit(self, X_train, y_train):
        # checking input data

        # get numpy array from different type
        X_train, y_train = get_numpy_array_train(X_train, y_train)

        # get float64 type
        X_train, y_train = convert_type_train(X_train, y_train)

        # y_train must have 1 dimension
        if len(y_train.shape) != 1:
            raise IndexError(f"Argument length 'X_train' != 'y_train': {X_train.shape[0]} != {len(y_train)}. "
                             f"Please, check input data.")

        # check lenght data
        if X_train.shape[0] != len(y_train):
            raise IndexError(f"Argument length 'X_train' != 'y_train': {X_train.shape[0]} != {len(y_train)}. "
                             f"Please, check input data.")

        # check n_combinations
        if self.n_combinations[0] > X_train.shape[1]:
            raise IndexError(f"Argument 'n_combinations' must be less than count features. "
                             f"n_combinations: {self.n_combinations[0]} > {X_train.shape[1]} count features")

        X_train, y_train = super(DecisionTreeAdaptive, self)._check_input(X_train, y_train)

        # fit tree
        self.tree = super(TreeRegressorAdaptive, self)._build(X_train, y_train, self.tree, self.max_depth)
        return self

    def predict(self, X_test):
        X_test = get_numpy_array_test(X_test)

        X_test = convert_type_test(X_test)

        X_test = super(DecisionTreeAdaptive, self)._check_input_test(X_test)

        return super(TreeRegressorAdaptive, self)._predict(X_test, self.tree)

    def test(self, X, index_tuples):
        return super(TreeRegressorAdaptive, self)._data_transform(X, index_tuples)