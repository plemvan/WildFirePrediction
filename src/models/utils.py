import numpy as np
import pandas as pd

def split_train_test(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    pd.DataFrame, pd.DataFrame: Training and testing DataFrames.
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return df.iloc[train_indices], df.iloc[test_indices]