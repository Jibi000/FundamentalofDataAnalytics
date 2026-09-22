import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
DATA_PATH = "/Users/jibithomas/Downloads/Sample_Superstore.csv"
df = pd.read_csv(DATA_PATH, encoding="latin1")
df_clean = df.copy()
desc_stats = df_clean.describe().round(2)

# ============================================================
# FIGURE 6: TOTAL SALES BY PRODUCT CATEGORY
# ============================================================

category_sales = (
    df_clean
    .groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))

sns.barplot(
    x=category_sales.index,
    y=category_sales.values
)

plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.show()
# ============================================================
# FIGURE 5: RELATIONSHIP BETWEEN SALES AND PROFIT
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_clean,
    x="Sales",
    y="Profit",
    alpha=0.5
)

plt.title("Relationship Between Sales and Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.tight_layout()
plt.show()
# ============================================================
# FIGURE 4: DISTRIBUTION OF SALES
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df_clean,
    x="Sales",
    bins=40,
    kde=True
)

plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
# ============================================================
# FIGURE 3: DESCRIPTIVE STATISTICS
# ============================================================

numerical_cols = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

desc_stats = df_clean[numerical_cols].describe().T

desc_stats["Median"] = df_clean[numerical_cols].median()

result = desc_stats[
    [
        "count",
        "mean",
        "Median",
        "std",
        "min",
        "max"
    ]
].round(2)

print("\nDescriptive Statistics:")
print(result)
