from tree import TreeRegressor, TreeRegressorSlow
from sklearn.datasets import fetch_california_housing
import time

data = fetch_california_housing()
X = data.data[:10000]
y = data.target[:10000]


model = TreeRegressor(max_depth = 10)
start = time.time()
model.fit(X, y)
end = time.time()
print(f"Fast time: {end - start} сек")

model_x = TreeRegressorSlow(max_depth = 10)
start = time.time()
model_x.fit(X, y)
end = time.time()
print(f"Slow time: {end - start} сек")

