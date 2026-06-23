import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

# ==================================
# Load Dataset
# ==================================
df = pd.read_csv("train.csv")

# ==================================
# Selected Features
# ==================================
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "FullBath",
    "BedroomAbvGr",
    "YearBuilt"
]

X = df[features]
y = df["SalePrice"]

# ==================================
# Train-Test Split
# ==================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==================================
# Train Model
# ==================================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==================================
# Predictions
# ==================================
predictions = model.predict(X_test)

# ==================================
# Evaluation Metrics
# ==================================
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")

# ==================================
# Cross Validation
# ==================================
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n===== CROSS VALIDATION =====")
print("Scores:", cv_scores)
print(f"Average CV R²: {cv_scores.mean():.4f}")

# ==================================
# Save Model
# ==================================
joblib.dump(model, "house_price_model.pkl")

print("\nModel saved as house_price_model.pkl")

# ==================================
# Create Images Folder
# ==================================
os.makedirs("images", exist_ok=True)

# ==================================
# Save Metrics CSV
# ==================================
metrics_df = pd.DataFrame({
    "Metric": [
        "R2 Score",
        "MAE",
        "RMSE",
        "Average CV R2"
    ],
    "Value": [
        r2,
        mae,
        rmse,
        cv_scores.mean()
    ]
})

metrics_df.to_csv(
    "images/model_metrics.csv",
    index=False
)

# ==================================
# Feature Importance
# ==================================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(feature_importance)

# Save Feature Importance CSV
feature_importance.to_csv(
    "images/feature_importance.csv",
    index=False
)

# ==================================
# Feature Importance Plot
# ==================================
plt.figure(figsize=(8, 5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)
plt.gca().invert_yaxis()

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    "images/feature_importance.png",
    bbox_inches="tight"
)

plt.close()

# ==================================
# Actual vs Predicted Plot
# ==================================
plt.figure(figsize=(8, 6))

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

plt.tight_layout()

plt.savefig(
    "images/actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.close()

print("\nPlots saved successfully!")