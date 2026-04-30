"""
dimensionality_reduction.py
---------------------------
PCA-based dimensionality reduction for the
Credit Card Customer Segmentation project.

Results
-------
* Components retained : 12
* Explained variance  : 99.52 %
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_COMPONENTS = 12          # components that preserve 99.52 % of variance
EXPLAINED_VARIANCE = 0.9952  # reference value from the study


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_pca(X_scaled: np.ndarray, n_components: int = N_COMPONENTS) -> tuple[np.ndarray, PCA]:
    """Fit PCA and project the scaled feature matrix.

    Parameters
    ----------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix (output of :func:`preprocessing.scale_features`).
    n_components : int, optional
        Number of principal components to retain. Default is 12, which
        preserves 99.52 % of the total variance in the study dataset.

    Returns
    -------
    X_pca : np.ndarray, shape (n_samples, n_components)
        PCA-transformed data.
    pca : PCA
        Fitted PCA object (exposes ``.explained_variance_ratio_`` etc.).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    total_var = pca.explained_variance_ratio_.sum()
    print(
        f"[fit_pca] {n_components} components retained → "
        f"{total_var * 100:.2f} % total explained variance."
    )
    return X_pca, pca


def select_n_components_by_variance(
    X_scaled: np.ndarray,
    threshold: float = 0.995,
) -> int:
    """Return the minimum number of PCA components needed to reach *threshold*.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    threshold : float
        Minimum cumulative explained variance to retain (default 0.995 ≈ 99.5 %).

    Returns
    -------
    int
        Number of components needed.
    """
    pca_full = PCA().fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n = int(np.searchsorted(cumvar, threshold)) + 1
    print(
        f"[select_n_components_by_variance] {n} components needed to reach "
        f"{threshold * 100:.1f} % explained variance."
    )
    return n


def plot_explained_variance(pca: PCA, save_path: str | None = None) -> None:
    """Plot cumulative explained variance vs. number of components.

    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    save_path : str or None
        If provided, the figure is saved to this path instead of displayed.
    """
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n = len(cumvar)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, n + 1), cumvar, marker="o", color="#4C72B0", linewidth=2)
    ax.axhline(99.52, color="tomato", linestyle="--", label="99.52% threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA – Cumulative Explained Variance")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_explained_variance] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
