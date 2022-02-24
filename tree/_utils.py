import pandas as pd
import numpy as np
from typing import List


def _check_param(max_depth, min_samples_leaf, min_samples_split):
    """Checking type parameters"""

    if not isinstance(max_depth, int):
        raise TypeError(f"Type 'max_depth' must be int. You send type {type(max_depth).__name__}")

    if not isinstance(min_samples_leaf, int):
        raise TypeError(f"Type 'min_samples_leaf' must be int. You send type {type(min_samples_leaf).__name__}")

    if not isinstance(min_samples_split, int):
        raise TypeError(f"Type 'min_samples_split' must be int. You send type {type(min_samples_split).__name__}")

def get_numpy_array(X, y):
    """Get np.ndarray from X, y"""

    # check X
    if isinstance(X, np.ndarray):
        pass
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(X, List):
        X = np.array(X)

    # check y
    if isinstance(y, np.ndarray):
        pass
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()
    if isinstance(y, List):
        y = np.array(y)

    if not isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame) and not isinstance(X, List):
        raise TypeError(f"Can't convert X, because X type is {type(X).__name__}. You must send Numpy, Pandas or List class object")

    if not isinstance(y, np.ndarray) and not isinstance(y, pd.DataFrame) and not isinstance(y, List):
        raise TypeError(f"Can't convert Y, because Y type is {type(y).__name__}. You must send Numpy, Pandas or List class object")

    return X, y