import numpy as np
cimport numpy as np
from tree import TreeRegressor
from tqdm import tqdm

cdef class LossFunction:
    def MSE(self, y, x):
        return np.square(y - x).mean()
    def MSE_der(self, y, x):
        return x - y



#------------------------------------------------------------------
#                      BASE - BOOSTING - CLASS
#------------------------------------------------------------------


cdef class BaseBoosting:
    def __init__(self,
                 base_model_class,
                 dict base_model_params,
                 int n_estimators,
                 float learning_rate,
                 bint randomization,
                 float subsample,
                 random_seed,
                 str custom_loss,
                 bint use_best_model,
                 n_iter_early_stopping,
                 float valid_control
                 ):

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # if true then bootstrap
        self.randomization = randomization
        self.subsample = subsample
        self.random_seed = random_seed
        self.custom_loss = custom_loss
        self.use_best_model = use_best_model
        self.n_iter_early_stopping = n_iter_early_stopping
        self.valid_control = valid_control

        self.loss = LossFunction()

        self.history = {}
        self.history['train'] = []
        self.history['valid'] = []

        self.models = []


        if self.custom_loss == 'mse':
            self.loss_fn = self.loss.MSE
            self.loss_derivative = self.loss.MSE_der

    cdef _base_build(self, _, sub_X, sub_y, predictions):
        """Building a tree base ensemble"""
        model = self.base_model_class(**self.base_model_params)

        if _ == 0:
            s_train = sub_y
        else:
            s_train = -self.loss_derivative(sub_y, predictions)


        model.fit(sub_X, s_train)

        predictions += self.learning_rate * model.predict(sub_X)

        self.models.append(model)

        return predictions

    cdef _append_history_without_valid(self, y_train, predictions):
        train_loss = self.loss_fn(y_train, predictions)
        self.history['train'].append(train_loss)

    cdef _append_history_with_valid(self, y_train, pred, y_valid, valid_pred):
        train_loss = self.loss_fn(y_train, pred)
        valid_loss = self.loss_fn(y_valid, valid_pred)
        self.history['train'].append(train_loss)
        self.history['valid'].append(valid_loss)



    cdef _predict_valid(self, _, X_valid, mean):
        valid_pred = np.ones([X_valid.shape[0]]) * mean

        for i in range(_):
            valid_pred += self.learning_rate * self.models[i].predict(X_valid)

        return valid_pred



    cdef get_bootstrap(self, sub_X, sub_y):
        np.random.seed(self.random_seed)

        ind = np.random.choice(np.arange(sub_X.shape[0]),
                           size=int(sub_X.shape[0] * self.subsample),
                           replace=False)
        sub_X_bootstrap, sub_y_bootstrap = sub_X[ind], sub_y[ind]

        return sub_X_bootstrap, sub_y_bootstrap



#------------------------------------------------------------------
#                      GRADIENT - BOOSTING - CLASS
#------------------------------------------------------------------


cdef class GradientBoostingRegressor(BaseBoosting):
    def __init__(self,
                 base_model_class = TreeRegressor,
                 dict base_model_params = None,
                 int n_estimators = 10,
                 float learning_rate = 1e-1,
                 bint randomization = False,
                 float subsample = 0.3,
                 random_seed = None,
                 str custom_loss = 'mse',
                 bint use_best_model = False,
                 n_iter_early_stopping = None,
                 valid_control = None):

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

    cdef fit_without_valid(self, np.ndarray X_train, np.ndarray y_train):
        self.mean_y_train = y_train.mean()

        if self.randomization:
            train_predictions = np.mean(y_train) * np.ones([int(self.subsample * y_train.shape[0])])
        else:
            train_predictions = np.mean(y_train) * np.ones([y_train.shape[0]])

        predictions = train_predictions.copy()

        X_train_, y_train_ = X_train, y_train

        for _ in tqdm(range(self.n_estimators)):

            if self.randomization:
                X_train_, y_train_ = self.get_bootstrap(X_train, y_train)

            predictions = self._base_build(_, X_train_, y_train_, predictions)

            if _ >= 1:
                self._append_history_without_valid(y_train_, predictions)




    cdef _fit_with_valid(self,
                         np.ndarray X_train,
                         np.ndarray y_train,
                         np.ndarray X_valid,
                         np.ndarray y_valid):

        self.mean_y_train = y_train.mean()

        mean = y_valid.mean()

        if self.randomization:
            train_predictions = np.mean(y_train) * np.ones([int(self.subsample * y_train.shape[0])])
        else:
            train_predictions = np.mean(y_train) * np.ones([y_train.shape[0]])

        predictions = train_predictions.copy()

        X_train_, y_train_ = X_train, y_train

        for _ in tqdm(range(self.n_estimators)):

            if self.randomization:
                X_train_, y_train_ = self.get_bootstrap(X_train, y_train)

            predictions = self._base_build(_, X_train_, y_train_, predictions)


            if _ >= 1:
                valid_predictions = self._predict_valid(_, X_valid, mean)

                self._append_history_with_valid(y_train_, predictions, y_valid, valid_predictions)

            if self.n_iter_early_stopping is not None and _ > self.n_iter_early_stopping:

                if abs(self.history['valid'][-1] - self.history['valid'][-self.n_iter_early_stopping]) <= self.valid_control:
                    self.n_estimators = _
                    break


    cpdef _build(self, np.ndarray X_train, np.ndarray y_train, X_valid = None, y_valid = None):
        if X_valid is None and y_valid is None:
            self.fit_without_valid(X_train, y_train)

        elif X_valid is not None and y_valid is not None:
            self._fit_with_valid(X_train, y_train, X_valid, y_valid)


    cdef _predict_best_model(self, X_test):
        arg_min = np.argmin(self.history['valid']) + 2

        predictions = np.ones([X_test.shape[0]]) * self.mean_y_train
        # predictions = np.zeros(X_test.shape[0])

        for _ in range(arg_min):
            predictions += self.learning_rate * self.models[_].predict(X_test)

        return predictions



    cpdef _predict(self, X_test):
        if self.use_best_model:
            return self._predict_best_model(X_test)
        else:
            predictions = np.ones([X_test.shape[0]]) * self.mean_y_train

            # predictions = np.zeros(X_test.shape[0])

            for _ in range(self.n_estimators):
                predictions += self.learning_rate * self.models[_].predict(X_test)

            return predictions









