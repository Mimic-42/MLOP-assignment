import os
import sys
import pandas as pd

# Add the src folder to the system path manually
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import preprocess

def test_preprocess_removes_nans():
    df = pd.DataFrame({
        "feature1": [1.0, None],
        "feature2": [2.0, 3.0]
    })
    processed = preprocess(df)
    assert processed.isnull().sum().sum() == 0

