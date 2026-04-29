"""
Utility functions for data splitting and preprocessing.

These helpers are designed to be importable without side effects —
no code runs at module import time.
"""

import numpy as np


def split_train_test(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to split.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default=0.2).
    random_state : int, optional
        Random seed for reproducibility (default=42).

    Returns
    -------
    train : pd.DataFrame
        Training subset.
    test : pd.DataFrame
        Testing subset.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": range(100), "b": range(100)})
    >>> train, test = split_train_test(df, test_size=0.2, random_state=0)
    >>> len(test) == 20
    True
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return df.iloc[train_indices], df.iloc[test_indices]
