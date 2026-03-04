"""
Worker Profiling — Clustering (K-Means / DBSCAN)
Identifies 3 distinct employee risk profiles and their top-3 features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.data.preprocessing import preprocess_data, create_composite_indices


CLUSTER_FEATURES = [
    "MH_Support_Index",
    "Stigma_Index",
    "Openness_Score",
    "Do you currently have a mental health disorder?",
    "Have you ever sought treatment for a mental health issue from a mental health professional?",
    "Do you have a family history of mental illness?",
    "Have you had a mental health disorder in the past?",
    "Do you believe your productivity is ever affected by a mental health issue?",
]


def find_optimal_k(X_scaled: np.ndarray, max_k: int = 8) -> int:
    """Use silhouette score to find best number of clusters."""
    scores = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return max(scores, key=scores.get)


def run_clustering(data_path: str = "data/raw/mental_health.csv", n_clusters: int = 3):
    df_raw = pd.read_csv(data_path)
    df_clean = preprocess_data(df_raw)
    df_features = create_composite_indices(df_clean)

    available = [c for c in CLUSTER_FEATURES if c in df_features.columns]
    X = df_features[available].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df_features["cluster"] = labels

    # Top-3 features per cluster by mean value
    cluster_profiles = {}
    for cluster_id in range(n_clusters):
        cluster_df = df_features[df_features["cluster"] == cluster_id][available]
        top3 = cluster_df.mean().sort_values(ascending=False).head(3).index.tolist()
        cluster_profiles[f"cluster_{cluster_id}"] = top3
        print(f"Cluster {cluster_id}: {top3}")

    return df_features, cluster_profiles


if __name__ == "__main__":
    run_clustering()
