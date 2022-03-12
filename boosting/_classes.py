from tree import TreeRegressor
from sklearn.base import BaseEstimator
from ._gradient_boosting import GradientBoostingRegressor




class GBRegressor(BaseEstimator, GradientBoostingRegressor):
    def __init__(self,
                 base_model_class = TreeRegressor,
                 base_model_params: dict = None,
                 n_estimators: int = 10,
                 learning_rate: float = 1e-3,
                 randomization = False,
                 subsample: float = 0.3,
                 random_seed: int = 42,
                 custom_loss: str = 'mse',
                 use_best_model: bool = False,
                 n_iter_early_stopping: int = None,
                 valid_control: float = 1e-10
                 ):

        super().__init__(base_model_class,
                         base_model_params,
                         n_estimators,
                         learning_rate,
                         randomization,
                         subsample,
                         random_seed,
                         custom_loss,
                         use_best_model,
                         n_iter_early_stopping,
                         valid_control)



    def fit(self, X_train, y_train, X_valid = None, y_valid = None):
        super()._build(X_train, y_train, X_valid, y_valid)
        return self

    def predict(self, X_test):
        return super()._predict(X_test)



