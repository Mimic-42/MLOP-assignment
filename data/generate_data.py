import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 200

# Two numerical features
feature1 = np.random.normal(0, 1, n_samples)
feature2 = np.random.normal(2, 1.5, n_samples)

# Binary target: 1 if sum of features > 2, else 0
target = (feature1 + feature2 > 2).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# Save to CSV
df.to_csv("raw.csv", index=False)
print("✅ Dataset saved as data/raw.csv")
