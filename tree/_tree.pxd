import numpy as np
cimport numpy as np
from sklearn.metrics import mean_squared_error as mse

cdef list _fit_tree(np.ndarray[np.float64_t, ndim = 2] sub_X, np.ndarray[np.float64_t, ndim = 1] sub_y, np.int64_t min_samples_leaf):
    cdef np.ndarray[np.int64_t, ndim = 1] arg
    cdef np.int64_t ind
    cdef np.float64_t N
    cdef np.float64_t Nl
    cdef np.float64_t Nr
    cdef np.int64_t thres
    cdef np.float64_t threshold
    cdef np.ndarray[np.float64_t, ndim = 1] target_left
    cdef np.ndarray[np.float64_t, ndim = 1] target_right
    cdef np.float64_t mean_left
    cdef np.float64_t mean_right

    cdef list mean_left_array
    cdef list mean_right_array

    cdef np.float64_t error_left
    cdef np.float64_t error_right

    cdef np.float64_t threshold_best

    cdef np.float64_t best_error

    value = sub_y.mean()

    best_error = ((sub_y - value) ** 2).sum()


    feature_split, left_value, right_value = None, None, None


    for feature in range(sub_X.shape[1]):
        feature_vector = sub_X[:, feature]

        # sort feature
        arg = np.argsort(feature_vector)

        # count samples in right and left child
        N = feature_vector.shape[0]
        Nl, Nr = N, 0
        thres = 1

        # start feature go
        while thres < N - 1:

            # add a sample in right and left child
            Nl -= 1
            Nr += 1

            # choose trashhold
            ind = arg[thres]
            threshold = feature_vector[ind]

            # skip similar feature
            ind = arg[thres + 1]
            if thres < N - 1 and threshold == feature_vector[ind]:
                thres += 1
                continue

            # the data that we get as a result of such a split
            target_left = sub_y[arg][thres:]
            target_right = sub_y[arg][:thres]
            mean_left = target_left.mean()
            mean_right = target_right.mean()

            mean_left_array = [mean_left for _ in range(target_left.shape[0])]
            mean_right_array = [mean_right for _ in range(target_right.shape[0])]

            error_left = (Nl / N) * mse(target_left, mean_left_array)
            error_right = (Nr / N) * mse(target_right, mean_right_array)


            if (error_left + error_right < best_error) and (min(Nl, Nr) > min_samples_leaf):
                threshold_best = threshold
                feature_split = feature
                left_value = mean_left
                right_value = mean_right

                best_error = error_left + error_right

            thres += 1

    return [value, threshold_best, feature_split, left_value, right_value]


cdef class TreeBuilder:
    """Parent class for different building tree strategies:"""

    cpdef _build(self, np.ndarray X_train, np.ndarray y_train, dict node, int depth):
        """Build a tree"""
        pass

    cpdef _get_predict_node(self, np.ndarray X_test, dict node):
        """Get predict"""
        pass

    cpdef _check_input(self, np.ndarray X, np.ndarray y):
        """Checking dtype X, y"""
        if X.dtype != np.float64 and X.dtype != np.float32 and X.dtype != np.int64 and X.dtype != np.int32:
            raise TypeError(f"X_train type must be int or float. Type X_train now: {X.dtype}")

        elif y.dtype != np.float64 and y.dtype != np.float32 and y.dtype != np.int64 and y.dtype != np.int32:
            raise TypeError(f"y_train type must be int or float. Type y_train now: {y.dtype}")

        return X, y

    cpdef _check_input_test(self, np.ndarray X):
        """Checking dtype X_test"""
        if X.dtype != np.float64 and X.dtype != np.float32 and X.dtype != np.int64 and X.dtype != np.int32:
            raise TypeError(f"X_test type must be int or float. Type X_test now: {X.dtype}")
        return X

cdef class DecisionTreeRegressorSlow(TreeBuilder):

    def __init__(self, int max_depth, int min_samples_leaf, int min_samples_split):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        
    cpdef _check_input(self, np.ndarray X, np.ndarray y):
        return super(TreeBuilder, self)._check_input(X, y)

    cpdef _check_input_test(self, np.ndarray X_test):
        return super(TreeBuilder, self)._check_input_test(X_test)

    cpdef _build(self, np.ndarray sub_X, np.ndarray sub_y, dict node, int depth):
        # check count samples for min_samples_split
        if len(sub_y) < self.min_samples_split:
            return

        # check depth
        if depth == 0:
            return

        # find opt split, value, threshold_best
        node['value'], node['threshold_best'], node['feature_split'], left_value, right_value = _fit_tree(sub_X,
                                                                                           sub_y,
                                                                                           self.min_samples_leaf)

        if node['feature_split'] is None:
            return

        node['left_child'], node['right_child'] = {}, {}

        node['left_child']['value'] = left_value
        node['left_child']['feature_split'] = None

        node['right_child']['value'] = right_value
        node['right_child']['feature_split'] = None


        idx_l = sub_X[:, node['feature_split']] > node['threshold_best']
        idx_r = sub_X[:, node['feature_split']] <= node['threshold_best']

        self._build(sub_X[idx_l, :], sub_y[idx_l], node['left_child'], depth - 1)
        self._build(sub_X[idx_r, :], sub_y[idx_r], node['right_child'], depth - 1)

        return node

    cpdef _get_predict_node(self, np.ndarray X_test, dict node):
        # return target if split not found - const value
        if node['feature_split'] is None:
            return node['value']

        # get down, if split found
        if X_test[node['feature_split']] > node['threshold_best']:
            return self._get_predict_node(X_test, node['left_child'])
        else:
            return self._get_predict_node(X_test, node['right_child'])

    cpdef _predict(self, np.ndarray X_test, dict node):
        """Get predict for X_test"""
        predict = []
        for obj in range(X_test.shape[0]):
            predict.append(self._get_predict_node(X_test[obj], node))
        return predict