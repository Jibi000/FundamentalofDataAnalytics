
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
DATA_PATH = "/Users/jibithomas/Downloads/Sample_Superstore.csv"
df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)
print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)
df_clean = df.copy()
df_clean["Order Date"] = pd.to_datetime(
    df_clean["Order Date"],
    errors="coerce"
)
df_clean["Ship Date"] = pd.to_datetime(
    df_clean["Ship Date"],
    errors="coerce"
)
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
print("Cleaned dataset shape:", df_clean.shape)
features = [
    "Quantity",
    "Discount",
    "Category",
    "Region",
    "Ship Mode"
]
target = "Sales"
model_df = df_clean[
    features + [target]
].copy()
model_df = model_df.dropna().reset_index(drop=True)
model_df = pd.get_dummies(
    model_df,
    columns=[
        "Category",
        "Region",
        "Ship Mode"
    ],
    drop_first=True
)
X = model_df.drop(
    columns=[target]
)
y = model_df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
# ============================================================
# MODEL 2: RANDOM FOREST REGRESSION
# ============================================================

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions
rf_predictions = random_forest_model.predict(X_test)

print("\nRandom Forest Regression")
print("First 10 predictions:")

for value in rf_predictions[:10]:
    print(round(value, 2))
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
print("\nLinear Regression")
print("First 10 predictions:")

for value in linear_predictions[:10]:
    print(round(value, 2))
print("\nTraining and testing data:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nNumber of observations:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nFeatures used:")
print(X.columns.tolist())