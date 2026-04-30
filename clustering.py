"""
clustering.py
-------------
Gaussian Mixture Model (GMM) clustering for the
Credit Card Customer Segmentation project.

Results
-------
* Optimal clusters  : 7
* Best Silhouette   : 0.46710
* Cluster range tested : 2 – 10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIMAL_N_CLUSTERS = 7
BEST_SILHOUETTE    = 0.46710
CLUSTER_RANGE      = range(2, 11)   # 2 to 10 inclusive

# Cluster labels and descriptions (used for profiling / reporting)
CLUSTER_PROFILES = {
    0: {
        "name"        : "Heavy Cash Advance Users",
        "type"        : "Credit-dependent / revolving balance customers",
        "key_features": ["High BALANCE", "High CASH_ADVANCE", "No installments",
                         "Low PRC_FULL_PAYMENT"],
        "insight"     : "Rely heavily on cash advances; higher financial risk.",
    },
    1: {
        "name"        : "Installment-Oriented Customers",
        "type"        : "Planned spenders using installment payments",
        "key_features": ["High INSTALLMENTS_PURCHASES", "No cash advances",
                         "Moderate balance", "Medium credit limit"],
        "insight"     : "Good candidates for installment-based promotions.",
    },
    2: {
        "name"        : "Cash + Installment Mixed Users",
        "type"        : "Financially stretched customers",
        "key_features": ["High cash advance", "High installment purchases",
                         "High balance", "Low full payment ratio"],
        "insight"     : "High revenue potential but also higher default risk.",
    },
    3: {
        "name"        : "Cash Advance Only Customers",
        "type"        : "Emergency cash users",
        "key_features": ["Almost no purchases", "High cash advance usage",
                         "Low transaction count"],
        "insight"     : "Consider risk monitoring and financial advisory services.",
    },
    4: {
        "name"        : "High Spenders (No Cash Advance)",
        "type"        : "Premium loyal customers",
        "key_features": ["Very high PURCHASES", "High INSTALLMENTS",
                         "High CREDIT_LIMIT", "No cash advance usage"],
        "insight"     : "Ideal for premium cards and loyalty rewards.",
    },
    5: {
        "name"        : "Super Active All-Channel Users",
        "type"        : "Very active, high-value customers",
        "key_features": ["High purchases", "High cash advances",
                         "High installments", "High payments"],
        "insight"     : "Important segment for retention and VIP programs.",
    },
    6: {
        "name"        : "Purchase-Only Customers",
        "type"        : "Regular transactional users",
        "key_features": ["Purchases only", "No cash advance",
                         "No installments", "Medium credit limit"],
        "insight"     : "Encourage installment usage to increase profitability.",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_optimal_clusters(
    X_pca: np.ndarray,
    cluster_range: range = CLUSTER_RANGE,
    random_state: int = 42,
) -> dict[int, float]:
    """Evaluate GMM for each cluster count and return silhouette scores.

    Parameters
    ----------
    X_pca : np.ndarray, shape (n_samples, n_components)
        PCA-projected data.
    cluster_range : range
        Range of cluster counts to test (default 2–10).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    dict[int, float]
        Mapping of ``{n_clusters: silhouette_score}``.
    """
    scores: dict[int, float] = {}
    for n in cluster_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=random_state,
        )
        labels = gmm.fit_predict(X_pca)
        score  = silhouette_score(X_pca, labels, sample_size=3000, random_state=random_state)
        scores[n] = score
        print(f"  n={n:2d}  silhouette={score:.5f}")

    best_n = max(scores, key=scores.get)
    print(
        f"\n[find_optimal_clusters] Best: n={best_n}, "
        f"silhouette={scores[best_n]:.5f}"
    )
    return scores


def fit_gmm(
    X_pca: np.ndarray,
    n_components: int = OPTIMAL_N_CLUSTERS,
    random_state: int = 42,
) -> tuple[np.ndarray, GaussianMixture]:
    """Fit the final Gaussian Mixture Model and return cluster labels.

    Parameters
    ----------
    X_pca : np.ndarray, shape (n_samples, n_pca_components)
        PCA-transformed data.
    n_components : int
        Number of mixture components (clusters). Default is 7.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Cluster assignment for each customer (0 to n_components-1).
    gmm : GaussianMixture
        Fitted model.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
    )
    labels = gmm.fit_predict(X_pca)
    sil    = silhouette_score(X_pca, labels, sample_size=3000, random_state=random_state)
    print(
        f"[fit_gmm] GMM fitted with {n_components} clusters. "
        f"Silhouette score = {sil:.5f}"
    )
    return labels, gmm


def plot_silhouette_scores(
    scores: dict[int, float],
    save_path: str | None = None,
) -> None:
    """Bar chart of silhouette scores for each tested cluster count.

    Parameters
    ----------
    scores : dict[int, float]
        Output of :func:`find_optimal_clusters`.
    save_path : str or None
        If provided, figure is saved here; otherwise displayed.
    """
    ns   = list(scores.keys())
    vals = list(scores.values())
    colors = ["#4C72B0" if n != OPTIMAL_N_CLUSTERS else "tomato" for n in ns]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ns, vals, color=colors, edgecolor="white", width=0.6)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("GMM – Silhouette Score vs. Number of Clusters")
    ax.set_xticks(ns)

    best_n = max(scores, key=scores.get)
    ax.annotate(
        f"Best: {scores[best_n]:.4f}",
        xy=(best_n, scores[best_n]),
        xytext=(best_n + 0.5, scores[best_n] + 0.005),
        fontsize=9,
        color="tomato",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_silhouette_scores] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def build_cluster_summary(df_clean: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute per-cluster mean statistics for the key financial features.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Original (pre-scaling) cleaned dataframe.
    labels : np.ndarray
        Cluster label per customer.

    Returns
    -------
    pd.DataFrame
        Pivot table with clusters as rows and mean feature values as columns.
        A ``CLUSTER_NAME`` column is prepended for readability.
    """
    df = df_clean.copy()
    df["CLUSTER"] = labels

    summary = df.groupby("CLUSTER").mean(numeric_only=True).round(2)
    summary.insert(0, "CLUSTER_NAME", [CLUSTER_PROFILES[i]["name"] for i in summary.index])
    return summary


def print_cluster_profiles(labels: np.ndarray) -> None:
    """Pretty-print the business profile for every cluster.

    Parameters
    ----------
    labels : np.ndarray
        Cluster assignment per customer.
    """
    unique, counts = np.unique(labels, return_counts=True)
    print("=" * 60)
    print("CLUSTER PROFILES")
    print("=" * 60)
    for cluster_id, count in zip(unique, counts):
        p = CLUSTER_PROFILES[cluster_id]
        pct = count / len(labels) * 100
        print(f"\nCluster {cluster_id} – {p['name']}  ({count:,} customers, {pct:.1f}%)")
        print(f"  Type    : {p['type']}")
        print(f"  Features: {', '.join(p['key_features'])}")
        print(f"  Insight : {p['insight']}")
    print("=" * 60)
