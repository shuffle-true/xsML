#=============================================================
#              Bagging and RandomForest modules
#=============================================================

import numpy as np
from tree import TreeRegressor
from tree import TreeRegressorSlow
from tree import TreeRegressorAdaptive

from sklearn.base import BaseEstimator

from ._utils import _check_base_model_class
from ._utils import _check_base_model_class_adaptive
from ._utils import _check_param



#=============================================================
#                  PARENT BAGGING CLASS
#=============================================================

class BaseBagging:
    def __init__(self,
                 base_model,
                 n_estimators,
                 max_depth,
                 min_samples_leaf,
                 min_samples_split,
                 adaptive = None,
                 n_combinations = None,
                 randomization = None):

        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.adaptive = adaptive
        self.n_combinations = n_combinations
        self.randomization = randomization


    def _append_regressor_list(self):
        if isinstance(self.base_model(), TreeRegressor):
            return [TreeRegressor(self.max_depth,
                                  self.min_samples_leaf,
                                  self.min_samples_split) for _ in range(self.n_estimators)]


        if isinstance(self.base_model(), TreeRegressorSlow):
            return [TreeRegressorSlow(self.max_depth,
                                      self.min_samples_leaf,
                                      self.min_samples_split) for _ in range(self.n_estimators)]

        if isinstance(self.base_model(), TreeRegressorAdaptive):

            return [TreeRegressorAdaptive(self.max_depth,
                                          self.min_samples_leaf,
                                          self.min_samples_split,
                                          self.adaptive,
                                          self.n_combinations,
                                          self.randomization) for _ in range(self.n_estimators)]



    def _get_bootstrap_data(self, X_train, y_train):
        ind = np.random.choice(np.arange(X_train.shape[0]),
                               size=X_train.shape[0],
                               replace=True)

        X_train_boot, y_train_boot = X_train[ind], y_train[ind]

        return X_train_boot, y_train_boot

    def _build(self, X_train, y_train, regressor_list):

        for i in range(self.n_estimators):
            X_train_boot, y_train_boot = self._get_bootstrap_data(X_train, y_train)

            regressor_list[i].fit(X_train_boot, y_train_boot)

        return regressor_list

    def _predict(self, X_test, regressor_list):
        predictions = []

        for i in range(self.n_estimators):
            predictions.append(regressor_list[i].predict(X_test))

        return np.array(np.mean(predictions, axis=0))



#=============================================================
#                  TREE BAGGING CLASS
#=============================================================

class BaggingTree(BaseEstimator, BaseBagging):
    def __init__(self,
                 base_model,
                 n_estimators,
                 max_depth,
                 min_samples_leaf,
                 min_samples_split):
        _check_base_model_class(base_model)
        _check_param(n_estimators)

        super().__init__(base_model,
                         n_estimators,
                         max_depth,
                         min_samples_leaf,
                         min_samples_split)

    def fit(self, X_train, y_train):
        self.regressor = self._append_regressor_list()
        self.regressor_fit = self._build(X_train, y_train, self.regressor)
        return self

    def predict(self, X_test):
        return self._predict(X_test, self.regressor_fit)


#=============================================================
#                  TREE ADAPTIVE BAGGING CLASS
#=============================================================

class BaggingTreeAdaptive(BaseEstimator, BaseBagging):
    def __init__(self,
                 base_model,
                 n_estimators,
                 max_depth,
                 min_samples_leaf,
                 min_samples_split,
                 adaptive=True,
                 n_combinations=2,
                 randomization='sum'):
        _check_base_model_class_adaptive(base_model)
        _check_param(n_estimators)

        super().__init__(base_model,
                         n_estimators,
                         max_depth,
                         min_samples_leaf,
                         min_samples_split,
                         adaptive,
                         n_combinations,
                         randomization)

    def fit(self, X_train, y_train):
        self.regressor = self._append_regressor_list()
        self.regressor_fit = self._build(X_train, y_train, self.regressor)
        return self

    def predict(self, X_test):
        return self._predict(X_test, self.regressor_fit)








