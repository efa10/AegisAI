"""
Tests for src.preprocessing.utils
"""

import pytest
import pandas as pd

from src.preprocessing.utils import split_data
from config.settings import SETTINGS

class TestSplitData:
    """Unit tests for the data splitting utility."""
    
    def test_splits_correct_ratio(self):
        """Should split data according to test_size."""
        # Create dummy data
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'label': [0] * 50 + [1] * 50
        })
        
        X_train, X_test, y_train, y_test = split_data(df, target_col='label', test_size=0.2)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
    def test_stratifies_target(self):
        """Should maintain class distribution."""
        df = pd.DataFrame({
            'feature1': range(100),
            'label': [0] * 80 + [1] * 20  # 80/20 class imbalance
        })
        
        _, _, y_train, y_test = split_data(df, target_col='label', test_size=0.2)
        
        # y_test should have exactly 20% of the class 1 samples (which is 4)
        assert (y_test == 1).sum() == 4
        # y_train should have exactly 80% of the class 1 samples (which is 16)
        assert (y_train == 1).sum() == 16

    def test_raises_error_missing_target(self):
        """Should raise ValueError if target column is missing."""
        df = pd.DataFrame({
            'feature1': range(10),
            'target': [0] * 5 + [1] * 5
        })
        
        with pytest.raises(ValueError, match="not found"):
            split_data(df, target_col='label')
