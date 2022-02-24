from tree import TreeRegressorSlow
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data[:10]
y = data.target[:11]


obj = TreeRegressorSlow()
print(obj.fit(X, y))
print(obj.predict(X))