import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Load data
df = pd.read_csv("../data/raw.csv")

# Split features and target
X = df[["feature1", "feature2"]]
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
# os.makedirs("src", exist_ok=True)
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved to src/model.pkl")
