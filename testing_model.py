from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tree import TreeRegressor
from tree import compute_biase_variance



data = fetch_california_housing()
X = data.data[:5000]
y = data.target[:5000]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)
model = TreeRegressor(max_depth = 3, min_samples_leaf = 5, min_samples_split = 9)
print("Bias: {0}, Varience: {1}, Error: {2}".format(*compute_biase_variance(model, X_train, y_train)))

