import numpy as np
from _best_split import find_best_split_classification
from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split
# np.random.seed(42)

class DecisionTreeClassifierReal(BaseEstimator):
    """
    Решающее дерево поддерживающее только вещественные признаки
    """
    def __init__(self, max_depth=np.inf, min_samples_split=0, min_samples_leaf=1):
        self.tree = {}
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        # сделаем проверку на один класс

        if np.all(sub_y == sub_y[0]):
            node['type'] = 'terminal'
            node['class'] = sub_y[0]
            return

        # проверка на нулевую глубину
        if depth == 0:
            node['type'] = 'terminal'
            node['class'] = sub_y[0]
            return

        # если кол-во объектов меньше чем заданный минимальный сплит также выходим
        if (len(sub_y) < self.min_samples_split) or (len(sub_y) < self.min_samples_leaf):
            node['type'] = 'terminal'
            node['class'] = sub_y[0]
            return

        # инициализируем параметры
        feature_best, threshold_best, gini_best, split = None, None, None, None

        #начинаем пробегаться по признакам
        for feature in range(sub_X.shape[1]):
            feature_vector = sub_X[:, feature]

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split_classification(feature_vector,
                                                    sub_y)

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold # возвращает True False в зависимости от условия

                threshold_best = threshold


        # Теперь напишем условие - если лучшая фича не определена, и если
        # минимальное число объектов одного класса в текущей вершине меньше,
        # чем минимально допустимое min_samples_leaf, то выбирается мажоритарный класс

        if feature_best is None or min(np.sum(split), np.sum(np.logical_not(split))) < self.min_samples_leaf:
            node['type'] = 'terminal'
            node['class'] = Counter(sub_y).most_common(1)[0][0]
            return


        # если это условие не выполняется, то идем дальше

        node['type'] = 'nonterminal'

        node['feature_split'] = feature_best
        node['threshold'] = threshold_best

        node['left_child'], node['right_child'] = {}, {}


        # рекурсивно вызываем для дочек, отправляя в левое поддерево только то, что
        # отделилось на этом участке, с правым поддерревом аналогично. Не забываем уменьшить порог

        self._fit_node(sub_X[split], sub_y[split], node['left_child'], depth - 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node['right_child'], depth - 1)


    def fit(self, X, y):
        self._fit_node(np.array(X), np.array(y), self.tree, self.max_depth)
        return self

    def _predict_node(self, x, node):
        """
        Если type = terminal, возвращаем единственный в листе класс

        Если нет, то рекурсивно вызываем для левого и правого поддерева и проделываем тоже самое
        """
        if node['type'] == 'terminal':
            return node['class']

        if (x[node['feature_split']] < node.get('threshold', -np.inf)):
            return self._predict_node(x, node['left_child'])
        return self._predict_node(x, node['right_child'])

    def predict(self, X):
        """
        Пробегаемся по всем объектам и для каждой строки вызываем _predict_node
        :param X:
        :return:
        """

        predicted = []
        for obj in np.array(X):
            predicted.append(self._predict_node(obj, self.tree))
        return np.array(predicted)


# dtc = DecisionTreeClassifierReal(max_depth = 5, min_samples_split = 2, min_samples_leaf = 1)
# X = np.ndarray((10000,2), buffer=np.random.normal(loc = 0,
#                                           scale=5,
#                                           size = (10000,2)))
#
# y = np.logical_xor(X[:, 0] > 0, X[:, 1] < 0).astype(int)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size = 0.2)
#
# dtc.fit(X_train, y_train)
# print(f'Accuracy = {acc(y_test, dtc.predict(X_test))}')



