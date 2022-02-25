from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tree import TreeRegressor, TreeRegressorAdaptive
from tree import compute_biase_variance
import numpy as np


data = fetch_california_housing()
X = data.data[:5]
y = data.target[:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)
model = TreeRegressorAdaptive(max_depth = 3, min_samples_leaf = 5, min_samples_split = 9)
# print("Bias: {0}, Varience: {1}, Error: {2}".format(*compute_biase_variance(model, X_train, y_train)))
print(model.test(X, [(0, 1), (0, 2), (0, 3)]))


def _data_transform(X, index_tuples):

    if 'sum' == 'sum' and 2 == 2:
        for i in index_tuples:
            X = np.hstack((X, X[:, (i[0], i[1])].sum(axis = 1).reshape(X.shape[0], 1)))
        return X

# print(_data_transform(X, [(0, 1), (0, 2), (0, 3)]))