from sklearn.datasets import fetch_california_housing
from tree import TreeRegressor
import pickle


data = fetch_california_housing()
X = data.data[:5]
y = data.target[:5]

model = TreeRegressor()
model.fit(X, y)
