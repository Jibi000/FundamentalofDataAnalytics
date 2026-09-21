# ==========================================
# TASK 4 - CLASSIFICATION AND CLUSTERING
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_score
)
from sklearn.cluster import KMeans


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
# DATA CLEANING
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

df_clean = df_clean.drop_duplicates().reset_index(drop=True)


# ==========================================
# CREATE PROCESSING DAYS
# ==========================================

df_clean["ProcessingDays"] = (
    df_clean["Ship Date"] -
    df_clean["Order Date"]
).dt.days


print("Cleaned dataset shape:", df_clean.shape)


# ==========================================
# CREATE CUSTOMER-LEVEL RFM DATA
# ==========================================

customer_df = df_clean.dropna(
    subset=["Customer ID"]
).copy()

reference_date = customer_df["Order Date"].max()


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
    rfm["Monetary"] /
    rfm["Frequency"]
)


print("\nRFM dataset shape:")
print(rfm.shape)

print("\nFirst five RFM records:")
print(rfm.head())


# ==================================================
# 4.1 CLASSIFICATION: PREDICTING VALUE SEGMENT
# ==================================================


# ==========================================
# CREATE VALUE SEGMENTS
# ==========================================

rfm["ValueSegment"] = pd.qcut(
    rfm["Monetary"],
    q=3,
    labels=[
        "Low",
        "Medium",
        "High"
    ],
    duplicates="drop"
)


print("\nValue Segment Distribution:")
print(
    rfm["ValueSegment"].value_counts()
    .sort_index()
)


# ==========================================
# FIGURE 15
# VALUE SEGMENT DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=rfm,
    x="ValueSegment",
    order=[
        "Low",
        "Medium",
        "High"
    ]
)

plt.title(
    "Figure 15: Value Segment Distribution After Quantile Binning"
)

plt.xlabel("Value Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# CLASSIFICATION FEATURES
# ==========================================

classification_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalProfit",
    "AverageDiscount"
]

X_class = rfm[
    classification_features
].copy()

y_class = rfm[
    "ValueSegment"
].copy()


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X_class,
    y_class,

    test_size=0.20,

    random_state=42,

    stratify=y_class
)


# ==========================================
# SCALE CLASSIFICATION FEATURES
# ==========================================

scaler_class = StandardScaler()

X_train_scaled = scaler_class.fit_transform(
    X_train
)

X_test_scaled = scaler_class.transform(
    X_test
)


# ==========================================
# LOGISTIC REGRESSION
# ==========================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


logistic_model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# PREDICTIONS
# ==========================================

y_pred = logistic_model.predict(
    X_test_scaled
)


# ==========================================
# CLASSIFICATION METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n====================================")
print("CLASSIFICATION RESULTS")
print("====================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ==========================================
# FIGURE 16
# CLASSIFICATION METRICS
# ==========================================

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})


print("\nFigure 16 - Classification Metrics:")
print(
    metrics_df.round(4)
)


# ==========================================
# DETAILED CLASSIFICATION REPORT
# ==========================================

print("\nDetailed Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# FIGURE 17
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)


plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Low",
        "Medium",
        "High"
    ]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Figure 17: Confusion Matrix for Logistic Regression"
)

plt.tight_layout()
plt.show()


# ==================================================
# 4.2 CLUSTERING: DISCOVERING CUSTOMER SEGMENTS
# ==================================================


# ==========================================
# CLUSTERING FEATURES
# ==========================================

clustering_features = [
    "Recency",
    "Frequency",
    "Monetary"
]


X_cluster = rfm[
    clustering_features
].copy()


# ==========================================
# STANDARDISE RFM VARIABLES
# ==========================================

scaler_cluster = StandardScaler()

X_cluster_scaled = scaler_cluster.fit_transform(
    X_cluster
)


# ==========================================
# FIGURE 18
# ELBOW METHOD
# ==========================================

inertia_values = []

k_values = range(2, 8)


for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(
        X_cluster_scaled
    )

    inertia_values.append(
        kmeans.inertia_
    )


plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title(
    "Figure 18: Elbow Curve for K-Means Clustering"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")

plt.xticks(
    list(k_values)
)

plt.tight_layout()
plt.show()


# ==========================================
# FIGURE 19
# SILHOUETTE SCORES
# ==========================================

silhouette_values = []


for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        X_cluster_scaled
    )

    score = silhouette_score(
        X_cluster_scaled,
        cluster_labels
    )

    silhouette_values.append(
        score
    )


silhouette_df = pd.DataFrame({

    "Number of Clusters": list(k_values),

    "Silhouette Score": silhouette_values

})


print("\nSilhouette Scores:")
print(
    silhouette_df.round(4)
)


plt.figure(figsize=(8, 5))

plt.plot(
    silhouette_df["Number of Clusters"],
    silhouette_df["Silhouette Score"],
    marker="o"
)

plt.title(
    "Figure 19: Silhouette Scores for k = 2 to 7"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")

plt.xticks(
    list(k_values)
)

plt.tight_layout()
plt.show()


# ==========================================
# SELECT FINAL NUMBER OF CLUSTERS
# ==========================================

# Select k with the highest silhouette score

best_k = silhouette_df.loc[
    silhouette_df["Silhouette Score"].idxmax(),
    "Number of Clusters"
]

best_k = int(best_k)


print("\nSelected number of clusters:", best_k)


# ==========================================
# FINAL K-MEANS MODEL
# ==========================================

kmeans_final = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)


rfm["Cluster"] = kmeans_final.fit_predict(
    X_cluster_scaled
)


# ==========================================
# CLUSTER MEANS
# ==========================================

cluster_means = rfm.groupby(
    "Cluster"
)[
    clustering_features
].mean()


print("\n====================================")
print("CLUSTER MEANS")
print("====================================")

print(
    cluster_means.round(2)
)


# ==========================================
# FIGURE 20
# CLUSTER MEANS
# ==========================================

cluster_means_plot = cluster_means.reset_index()

cluster_means_melted = cluster_means_plot.melt(
    id_vars="Cluster",
    value_vars=[
        "Recency",
        "Frequency",
        "Monetary"
    ],
    var_name="Variable",
    value_name="Mean"
)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=cluster_means_melted,
    x="Variable",
    y="Mean",
    hue="Cluster"
)

plt.title(
    "Figure 20: Cluster Means for Recency, Frequency and Monetary Value"
)

plt.xlabel("RFM Variable")
plt.ylabel("Mean Value")

plt.tight_layout()
plt.show()


# ==========================================
# FIGURE 21
# CUSTOMER SEGMENTS
# ==========================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="tab10",
    s=70,
    alpha=0.7
)

plt.title(
    "Figure 21: Customer Segments by Frequency and Monetary Value"
)

plt.xlabel("Frequency")
plt.ylabel("Monetary Value")

plt.tight_layout()
plt.show()


# ==========================================
# FINAL CLUSTER SUMMARY
# ==========================================

print("\n====================================")
print("FINAL CUSTOMER SEGMENT SUMMARY")
print("====================================")

print(
    rfm.groupby("Cluster")[
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    ].mean().round(2)
)