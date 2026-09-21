import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
DATA_PATH = "/Users/jibithomas/Downloads/Sample_Superstore.csv"
df = pd.read_csv(DATA_PATH, encoding="latin1")
df_clean = df.copy()
df_clean["Order Date"] = pd.to_datetime(
    df_clean["Order Date"],
    errors="coerce"
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cityblock

# Features used for customer segmentation
rfm_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalProfit",
    "AverageDiscount"
]

# --------------------------------------------------
# RFM Analysis
# --------------------------------------------------

customer_df = df_clean.dropna(subset=["Customer ID"]).copy()

reference_date = customer_df["Order Date"].max()

rfm = customer_df.groupby("Customer ID").agg(
    Recency=(
        "Order Date",
        lambda x: (reference_date - x.max()).days + 1
    ),
    Frequency=(
        "Order ID",
        "nunique"
    ),
    Monetary=(
        "Sales",
        "sum"
    ),
    TotalProfit=(
        "Profit",
        "sum"
    ),
    AverageDiscount=(
        "Discount",
        "mean"
    )
).reset_index()

# Average order value
rfm["AverageOrderValue"] = (
    rfm["Monetary"] / rfm["Frequency"]
)

print("\nRFM columns:")
print(rfm.columns.tolist())

print("\nRFM dataset:")
print(rfm.head())


# --------------------------------------------------
# Scaling
# --------------------------------------------------

from sklearn.preprocessing import StandardScaler

rfm_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalProfit",
    "AverageDiscount"
]

# Check that all required columns exist
print("\nFeatures used for scaling:")
print(rfm_features)

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm[rfm_features]
)

rfm_scaled_df = pd.DataFrame(
    rfm_scaled,
    columns=rfm_features
)

print("\nStandardised RFM features:")
print(rfm_scaled_df.head())

# Euclidean distance between first two customers
euclidean_distance = euclidean_distances(
    rfm_scaled_df.iloc[[0]],
    rfm_scaled_df.iloc[[1]]
)[0][0]

# Manhattan distance between first two customers
manhattan_distance = cityblock(
    rfm_scaled_df.iloc[0],
    rfm_scaled_df.iloc[1]
)

print("Euclidean distance:",
      round(euclidean_distance, 3))

print("Manhattan distance:",
      round(manhattan_distance, 3))



    