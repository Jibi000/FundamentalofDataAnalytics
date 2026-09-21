# ==========================================
# TASK 5 - VISUALISATION AND BUSINESS INSIGHTS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# LOAD DATA
# ==========================================

DATA_PATH = "/Users/jibithomas/Downloads/Sample_Superstore.csv"

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

print("Original dataset shape:", df.shape)


# ==========================================
# BASIC CLEANING
# ==========================================

df_clean = df.copy()

df_clean["Order Date"] = pd.to_datetime(
    df_clean["Order Date"],
    errors="coerce"
)

df_clean = df_clean.drop_duplicates().reset_index(drop=True)

print("Cleaned dataset shape:", df_clean.shape)


# ==================================================
# 5.1 REVENUE PATTERNS
# ==================================================


# ==========================================
# MONTHLY REVENUE
# ==========================================

monthly_revenue = (
    df_clean
    .dropna(subset=["Order Date"])
    .groupby(
        df_clean.dropna(subset=["Order Date"])["Order Date"]
        .dt.to_period("M")
    )["Sales"]
    .sum()
    .reset_index()
)

monthly_revenue.columns = [
    "Month",
    "Revenue"
]

monthly_revenue["Month"] = (
    monthly_revenue["Month"]
    .astype(str)
)


print("\nMonthly Revenue:")
print(monthly_revenue)


# ==========================================
# FIGURE 22
# MONTHLY REVENUE OVER TIME
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_revenue["Month"],
    monthly_revenue["Revenue"],
    marker="o"
)

plt.title(
    "Figure 22: Monthly Revenue Over Time"
)

plt.xlabel("Month")
plt.ylabel("Revenue")

plt.xticks(
    rotation=45
)

plt.tight_layout()
plt.show()


# ==========================================
# TOP 10 PRODUCTS
# ==========================================

top_products = (
    df_clean
    .groupby("Product Name")["Sales"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
    .sort_values()
)


print("\nTop 10 Products by Revenue:")
print(top_products)


# ==========================================
# FIGURE 23
# TOP TEN PRODUCTS
# ==========================================

plt.figure(figsize=(10, 6))

top_products.plot(
    kind="barh"
)

plt.title(
    "Figure 23: Top Ten Products by Total Revenue"
)

plt.xlabel("Total Revenue")
plt.ylabel("Product")

plt.tight_layout()
plt.show()


# ==========================================
# TOP 10 COUNTRIES
# ==========================================

top_countries = (
    df_clean
    .groupby("Country")["Sales"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
    .sort_values()
)


print("\nTop 10 Countries by Revenue:")
print(top_countries)


# ==========================================
# FIGURE 24
# TOP TEN COUNTRIES
# ==========================================

plt.figure(figsize=(9, 6))

top_countries.plot(
    kind="barh"
)

plt.title(
    "Figure 24: Top Ten Countries by Total Revenue"
)

plt.xlabel("Total Revenue")
plt.ylabel("Country")

plt.tight_layout()
plt.show()


# ==================================================
# 5.2 CUSTOMER VALUE DISTRIBUTION
# ==================================================


# ==========================================
# CREATE CUSTOMER MONETARY VALUE
# ==========================================

customer_monetary = (
    df_clean
    .dropna(subset=["Customer ID"])
    .groupby("Customer ID")["Sales"]
    .sum()
    .reset_index()
)


customer_monetary.columns = [
    "Customer ID",
    "Monetary"
]


print("\nCustomer Monetary Value:")
print(
    customer_monetary.head()
)


print(
    "\nCustomer Monetary Value Statistics:"
)

print(
    customer_monetary["Monetary"]
    .describe()
)


# ==========================================
# FIGURE 25
# CUSTOMER MONETARY VALUE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    customer_monetary["Monetary"],
    bins=30,
    kde=True
)

plt.title(
    "Figure 25: Distribution of Customer Monetary Value"
)

plt.xlabel("Customer Monetary Value")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# CREATE RFM DATA
# ==========================================

customer_df = df_clean.dropna(
    subset=["Customer ID"]
).copy()


reference_date = customer_df[
    "Order Date"
].max()


rfm = customer_df.groupby(
    "Customer ID"
).agg(

    Recency=(
        "Order Date",
        lambda x:
        (reference_date - x.max()).days + 1
    ),

    Frequency=(
        "Order ID",
        "nunique"
    ),

    Monetary=(
        "Sales",
        "sum"
    )

).reset_index()


print("\nRFM dataset:")
print(
    rfm.head()
)


# ==========================================
# CORRELATION MATRIX
# ==========================================

rfm_features = [
    "Recency",
    "Frequency",
    "Monetary"
]


correlation_matrix = rfm[
    rfm_features
].corr()


print("\nRFM Correlation Matrix:")
print(
    correlation_matrix.round(3)
)


# ==========================================
# FIGURE 26
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title(
    "Figure 26: Correlation Heatmap of RFM Features"
)

plt.tight_layout()
plt.show()


# ==========================================
# FINAL OUTPUT FOR REPORT
# ==========================================

print("\n====================================")
print("TASK 5 SUMMARY")
print("====================================")

print("\nMonthly revenue:")
print(monthly_revenue.head())

print("\nTop 10 products:")
print(top_products.sort_values(ascending=False))

print("\nTop 10 countries:")
print(top_countries.sort_values(ascending=False))

print("\nCustomer monetary value:")
print(
    customer_monetary["Monetary"].describe()
)

print("\nRFM correlations:")
print(
    correlation_matrix.round(3)
)