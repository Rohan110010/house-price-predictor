import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("train.csv")

# -----------------------------
# Select Important Features
# -----------------------------
X = df[
    [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "FullBath",
        "BedroomAbvGr",
        "YearBuilt"
    ]
]

# Target Variable
y = df["SalePrice"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Train Random Forest Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Make Predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "house_price_model.pkl")

print("Model trained successfully!")

# -----------------------------
# Create Images Folder
# -----------------------------
os.makedirs("images", exist_ok=True)

# -----------------------------
# Actual vs Predicted Plot
# -----------------------------
plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.savefig("images/actual_vs_predicted.png")

plt.show()