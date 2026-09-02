"""
AegisAI — Preprocessing Utilities

Reusable functions for data splitting, validation, and transformations.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from config.settings import SETTINGS

def split_data(
    df: pd.DataFrame, 
    target_col: str, 
    test_size: float = SETTINGS.TEST_SIZE, 
    random_state: int = SETTINGS.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into train and test sets with stratification.
    
    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    target_col : str
        The column name of the target variable for stratification.
    test_size : float
        The proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducible splits.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test
