import numpy as np

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    R = len(feature_vector)
    arg = feature_vector.argsort()
    target = np.array(target_vector)[arg]
    feature = np.array(feature_vector)[arg]
    thresholds = (feature[1:] + feature[:-1]) / 2
    nl = np.cumsum(target)[:-1]
    nr = target.sum() - nl
    Rl = np.arange(1, R)
    Rr = R - Rl
    pl = nl / Rl
    pr = nr / Rr
    Hl = 1 - pl ** 2 - (1 - pl) ** 2
    Hr = 1 - pr ** 2 - (1 - pr) ** 2
    ginis = -(Rr / R) * Hr - (Rl / R) * Hl
    unique = 1 - np.isin(thresholds, feature)
    thresholds = thresholds[np.where(unique)]
    ginis = ginis[np.where(unique)]
    opt = ginis.argmax()
    threshold_best = thresholds[opt]
    gini_best = ginis[opt]
    return thresholds, ginis, threshold_best, gini_best