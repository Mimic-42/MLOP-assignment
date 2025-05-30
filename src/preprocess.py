import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Simple example: fill any missing values with zero
    return df.fillna(0)
