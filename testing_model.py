import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


from tree._classes import TreeRegressor


df = pd.read_csv('DataFrame_after_preprocessing.csv')
df = df.drop(df.columns[21:], axis = 1)
df.price = df.price.apply(lambda x: x.replace(' ', ''))
df = df[df.price.astype(int) < 800000]
df = df[df.price.astype(int) > 1000]
df = df.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(df.drop(['price'], axis = 1), df['price'], random_state = 42, train_size = 0.8)
model = TreeRegressor(max_depth=10)
model.fit(X_train, y_train)
model.predict(X_test)
print('ok')