"""
Task 3: Regression Modelling, Optimisation and Evaluation
Predicts customer Monetary value (total Sales) from RFM-style behavioural features.
Self-contained: loads the raw CSV, cleans it, builds the customer-level table,
then splits, trains and evaluates the models. Run this file on its own.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Load the raw data
# ---------------------------------------------------------------------------
# EDIT THIS PATH if your CSV lives somewhere else:
DATA_PATH = "/Users/jibithomas/Downloads/Sample_Superstore.csv"


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Update DATA_PATH at the top of this script "
        "to point at your Sample_Superstore.csv."
    )

df = pd.read_csv(DATA_PATH, encoding="latin1")
print("Raw shape:", df.shape)

# ---------------------------------------------------------------------------
# 2. Clean and engineer features (mirrors task1.py / task2.py)
# ---------------------------------------------------------------------------
df_clean = df.copy()
df_clean["Order Date"] = pd.to_datetime(df_clean["Order Date"], errors="coerce")
df_clean["Ship Date"] = pd.to_datetime(df_clean["Ship Date"], errors="coerce")
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

df_clean["ProcessingDays"] = (df_clean["Ship Date"] - df_clean["Order Date"]).dt.days
df_clean["OrderYear"] = df_clean["Order Date"].dt.year
df_clean["OrderMonth"] = df_clean["Order Date"].dt.month

print("Cleaned shape:", df_clean.shape)

# ---------------------------------------------------------------------------
# 3. Build the customer-level RFM table
# ---------------------------------------------------------------------------
snapshot_date = df_clean["Order Date"].max() + pd.Timedelta(days=1)

customer_df = df_clean.groupby("Customer ID").agg(
    LastOrder=("Order Date", "max"),
    Frequency=("Order ID", "nunique"),
    Monetary=("Sales", "sum"),
    AvgBasketValue=("Sales", "mean"),
    TotalProfit=("Profit", "sum"),
    Region=("Region", lambda s: s.mode().iloc[0]),
).reset_index()
customer_df["Recency"] = (snapshot_date - customer_df["LastOrder"]).dt.days

print("Customer-level shape:", customer_df.shape)

# ---------------------------------------------------------------------------
# 4. Build X (features) and y (target)
# ---------------------------------------------------------------------------
region_dummies = pd.get_dummies(customer_df["Region"], prefix="Region", drop_first=True).astype(int)
customer_df = pd.concat([customer_df, region_dummies], axis=1)

FEATURES = ["Recency", "Frequency", "AvgBasketValue"] + list(region_dummies.columns)
X = customer_df[FEATURES].astype(float)
y = customer_df["Monetary"]

print("X shape:", X.shape, "| y shape:", y.shape)

# ---------------------------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

# ---------------------------------------------------------------------------
# 6. Fit models
# ---------------------------------------------------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# ---------------------------------------------------------------------------
# FIGURE 12: MODEL PERFORMANCE COMPARISON
# ---------------------------------------------------------------------------
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest Regression"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R²": [linear_r2, rf_r2],
})

print("\nModel Performance Comparison:")
print(results.round(2))
# FIGURE 13: ACTUAL VS PREDICTED SALES

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    linear_predictions,
    alpha=0.5,
    label="Linear Regression"
)

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.5,
    label="Random Forest"
)

# Reference line
min_value = min(y_test.min(), linear_predictions.min(), rf_predictions.min())
max_value = max(y_test.max(), linear_predictions.max(), rf_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Actual vs predicted plot (Figure 14 equivalent)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, rf_predictions, alpha=0.6)
lims = [0, max(y_test.max(), rf_predictions.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Actual Monetary value ($)")
ax.set_ylabel("Predicted Monetary value ($)")
ax.set_title("Actual vs Predicted: Random Forest")
ax.legend()
plt.tight_layout()
plt.savefig("figure14_actual_vs_predicted.png", dpi=150)
plt.show()