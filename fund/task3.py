# ==========================================
# TASK 3 - REGRESSION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "/Users/jibithomas/Downloads/Sample_Superstore.csv"

df = pd.read_csv(DATA_PATH, encoding="cp1252")

print("Original dataset shape:", df.shape)


# ==========================================
# 2. DATA CLEANING
# ==========================================

df_clean = df.copy()

df_clean["Order Date"] = pd.to_datetime(
    df_clean["Order Date"],
    errors="coerce"
)

df_clean["Ship Date"] = pd.to_datetime(
    df_clean["Ship Date"],
    errors="coerce"
)

# Remove duplicate rows
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# Create processing time
df_clean["ProcessingDays"] = (
    df_clean["Ship Date"] - df_clean["Order Date"]
).dt.days

print("Cleaned dataset shape:", df_clean.shape)


# ==========================================
# 3.1 TARGET AND PREDICTOR SELECTION
# ==========================================

features = [
    "Quantity",
    "Discount",
    "ProcessingDays",
    "Category",
    "Region",
    "Ship Mode"
]

target = "Sales"


# Keep only required columns
model_df = df_clean[features + [target]].copy()

# Remove rows with missing values
model_df = model_df.dropna().reset_index(drop=True)


print("\nModel dataset shape:", model_df.shape)
print("\nSelected predictors:")
print(features)

print("\nTarget variable:")
print(target)


# ==========================================
# 3. ENCODE CATEGORICAL VARIABLES
# ==========================================

model_df = pd.get_dummies(
    model_df,
    columns=[
        "Category",
        "Region",
        "Ship Mode"
    ],
    drop_first=True
)


# ==========================================
# CREATE X AND y
# ==========================================

X = model_df.drop(columns=[target])
y = model_df[target]

print("\nFeature matrix X shape:", X.shape)
print("Target y shape:", y.shape)

print("\nFeatures used:")
print(X.columns.tolist())


# ==========================================
# TRAIN-TEST SPLITS
# ==========================================

# 80/20 split
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 70/30 split
X_train_70, X_test_70, y_train_70, y_test_70 = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


print("\n====================================")
print("80/20 TRAIN-TEST SPLIT")
print("====================================")

print("X_train:", X_train_80.shape)
print("X_test :", X_test_80.shape)
print("y_train:", y_train_80.shape)
print("y_test :", y_test_80.shape)


print("\n====================================")
print("70/30 TRAIN-TEST SPLIT")
print("====================================")

print("X_train:", X_train_70.shape)
print("X_test :", X_test_70.shape)
print("y_train:", y_train_70.shape)
print("y_test :", y_test_70.shape)


# ==========================================
# FIGURE 11
# Train-Test Split Dimensions
# ==========================================

split_results = pd.DataFrame({
    "Split": [
        "80/20",
        "80/20",
        "70/30",
        "70/30"
    ],

    "Dataset": [
        "Training",
        "Testing",
        "Training",
        "Testing"
    ],

    "Rows": [
        len(X_train_80),
        len(X_test_80),
        len(X_train_70),
        len(X_test_70)
    ]
})


plt.figure(figsize=(8, 5))

sns.barplot(
    data=split_results,
    x="Split",
    y="Rows",
    hue="Dataset"
)

plt.title("Figure 11: Train-Test Split Dimensions")
plt.xlabel("Split Configuration")
plt.ylabel("Number of Rows")

plt.tight_layout()
plt.show()


# ==========================================
# FUNCTION FOR MODEL EVALUATION
# ==========================================

def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    split_name
):

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
    }

    results = []

    for name, model in models.items():

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({

            "Split": split_name,

            "Model": name,

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2
        })

    return pd.DataFrame(results)


# ==========================================
# MODEL RESULTS - 80/20
# ==========================================

results_80 = evaluate_models(
    X_train_80,
    X_test_80,
    y_train_80,
    y_test_80,
    "80/20"
)


# ==========================================
# MODEL RESULTS - 70/30
# ==========================================

results_70 = evaluate_models(
    X_train_70,
    X_test_70,
    y_train_70,
    y_test_70,
    "70/30"
)


# Combine results

results = pd.concat(
    [
        results_80,
        results_70
    ],
    ignore_index=True
)


# ==========================================
# 3.2 MODEL RESULTS
# ==========================================

print("\n====================================")
print("REGRESSION RESULTS")
print("====================================")

print(results.round(3))


# ==========================================
# FIGURE 12
# 80/20 MODEL COMPARISON
# ==========================================

results_80_plot = results_80.melt(
    id_vars=["Model"],
    value_vars=[
        "MAE",
        "RMSE",
        "R2"
    ],
    var_name="Metric",
    value_name="Value"
)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_80_plot,
    x="Metric",
    y="Value",
    hue="Model"
)

plt.title(
    "Figure 12: Regression Model Comparison - 80/20 Split"
)

plt.xlabel("Evaluation Metric")
plt.ylabel("Value")

plt.tight_layout()
plt.show()


# ==========================================
# FIGURE 13
# 70/30 MODEL COMPARISON
# ==========================================

results_70_plot = results_70.melt(
    id_vars=["Model"],
    value_vars=[
        "MAE",
        "RMSE",
        "R2"
    ],
    var_name="Metric",
    value_name="Value"
)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_70_plot,
    x="Metric",
    y="Value",
    hue="Model"
)

plt.title(
    "Figure 13: Regression Model Comparison - 70/30 Split"
)

plt.xlabel("Evaluation Metric")
plt.ylabel("Value")

plt.tight_layout()
plt.show()


# ==========================================
# FIGURE 14
# ACTUAL VS PREDICTED SALES
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_80,
    y_train_80
)

rf_predictions = rf_model.predict(
    X_test_80
)


plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_80,
    rf_predictions,
    alpha=0.5
)

min_value = min(
    y_test_80.min(),
    rf_predictions.min()
)

max_value = max(
    y_test_80.max(),
    rf_predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title(
    "Figure 14: Actual vs Predicted Sales - Random Forest Regressor"
)

plt.tight_layout()
plt.show()


# ==========================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        rf_model.feature_importances_
})


feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)


print("\n====================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("====================================")

print(
    feature_importance.head(10).round(4)
)

