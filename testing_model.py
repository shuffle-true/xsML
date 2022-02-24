from tree import TreeRegressor, TreeRegressorSlow
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error as mse

data = fetch_california_housing()
X = data.data[:15000]
y = data.target[:15000]


model = TreeRegressor(max_depth = 3, min_samples_leaf=10)
model.fit(X, y)
print(mse(y, model.predict(X)))




