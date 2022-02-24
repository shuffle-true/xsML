from tree import TreeRegressorSlow
from tree import compute_biase_variance
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data[:100]
y = data.target[:100]


model = TreeRegressorSlow(max_depth = 10)
print(compute_biase_variance(model, X, y, num_runs = 50))

