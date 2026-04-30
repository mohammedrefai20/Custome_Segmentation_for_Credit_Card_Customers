"""
visualization.py
----------------
Plotting utilities for the Credit Card Customer Segmentation project.

Functions
---------
* plot_feature_distributions  – histograms before / after log transform
* plot_tsne                   – 2-D t-SNE cluster scatter
* plot_cluster_heatmap        – cluster × feature mean heatmap
* plot_cluster_distribution   – bar chart of cluster sizes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Palette – one colour per cluster (up to 10)
# ---------------------------------------------------------------------------

CLUSTER_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    features: list[str],
    save_path: str | None = None,
) -> None:
    """Side-by-side histograms comparing distributions before and after log transform.

    Parameters
    ----------
    df_before : pd.DataFrame
        Dataset before log transformation.
    df_after : pd.DataFrame
        Dataset after log transformation.
    features : list[str]
        Feature columns to plot.
    save_path : str or None
        File path to save the figure, or None to display it.
    """
    n = len(features)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))

    for i, feat in enumerate(features):
        ax_before, ax_after = axes[i]

        df_before[feat].dropna().hist(ax=ax_before, bins=50, color="#4C72B0", edgecolor="white")
        ax_before.set_title(f"{feat} – Before", fontsize=9)
        ax_before.set_xlabel("")

        df_after[feat].dropna().hist(ax=ax_after, bins=50, color="#55A868", edgecolor="white")
        ax_after.set_title(f"{feat} – After Log Transform", fontsize=9)
        ax_after.set_xlabel("")

    fig.suptitle("Feature Distributions: Before vs After Log Transformation", fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_feature_distributions] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_tsne(
    X_pca: np.ndarray,
    labels: np.ndarray,
    cluster_names: dict[int, str] | None = None,
    perplexity: int = 40,
    n_iter: int = 1000,
    random_state: int = 42,
    save_path: str | None = None,
) -> None:
    """2-D t-SNE scatter plot coloured by cluster label.

    Parameters
    ----------
    X_pca : np.ndarray, shape (n_samples, n_pca_components)
        PCA-projected data.
    labels : np.ndarray, shape (n_samples,)
        Cluster assignments.
    cluster_names : dict or None
        Optional mapping of ``{cluster_id: display_name}``.
    perplexity : int
        t-SNE perplexity (default 40).
    n_iter : int
        t-SNE iterations (default 1 000).
    random_state : int
        Reproducibility seed.
    save_path : str or None
        Save path or None to display.
    """
    print("[plot_tsne] Running t-SNE (this may take a minute) …")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(X_pca)

    unique_labels = np.unique(labels)
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = cluster_names.get(lbl, f"Cluster {lbl}") if cluster_names else f"Cluster {lbl}"
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=8,
            alpha=0.6,
            color=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)],
            label=f"{lbl}: {name}",
        )

    ax.set_title("t-SNE Visualisation of Customer Clusters", fontsize=13)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_tsne] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_cluster_heatmap(
    cluster_summary: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Heatmap of normalised mean feature values per cluster.

    Parameters
    ----------
    cluster_summary : pd.DataFrame
        Output of :func:`clustering.build_cluster_summary`.
    save_path : str or None
        Save path or None to display.
    """
    # Drop the text label column for numeric processing
    numeric_cols = cluster_summary.select_dtypes(include=np.number).columns
    data = cluster_summary[numeric_cols]

    # Normalise column-wise so every feature is in [0, 1]
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)

    row_labels = [
        f"{idx}: {cluster_summary.loc[idx, 'CLUSTER_NAME']}"
        if "CLUSTER_NAME" in cluster_summary.columns
        else str(idx)
        for idx in data_norm.index
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        data_norm,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        yticklabels=row_labels,
        annot=False,
    )
    ax.set_title("Cluster Profiles – Normalised Mean Feature Values", fontsize=12)
    ax.set_xlabel("Features")
    ax.set_ylabel("Cluster")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_cluster_heatmap] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_cluster_distribution(
    labels: np.ndarray,
    cluster_names: dict[int, str] | None = None,
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart showing the size of each cluster.

    Parameters
    ----------
    labels : np.ndarray
        Cluster assignments.
    cluster_names : dict or None
        Optional ``{cluster_id: name}`` mapping.
    save_path : str or None
        Save path or None to display.
    """
    unique, counts = np.unique(labels, return_counts=True)
    bar_labels = [
        f"{lbl}: {cluster_names[lbl]}" if cluster_names else f"Cluster {lbl}"
        for lbl in unique
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        bar_labels,
        counts,
        color=[CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(len(unique))],
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
    ax.set_xlabel("Number of Customers")
    ax.set_title("Customer Distribution Across Clusters", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_cluster_distribution] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
