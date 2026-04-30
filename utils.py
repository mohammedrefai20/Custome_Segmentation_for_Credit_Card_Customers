"""
utils.py
--------
Miscellaneous helper utilities for the
Credit Card Customer Segmentation project.

Includes
--------
* save_results       – persist labelled dataframe to CSV
* save_model         – pickle GMM and PCA objects
* load_model         – restore pickled objects
* set_plot_style     – global Matplotlib / Seaborn aesthetics
* print_section      – pretty section header for console output
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_results(df_clean: pd.DataFrame, labels: np.ndarray, path: str = "outputs/segmented_customers.csv") -> None:
    """Attach cluster labels to the cleaned dataframe and write to CSV.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned (un-scaled) customer data.
    labels : np.ndarray
        Cluster label per customer.
    path : str
        Output file path. Parent directories are created if absent.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    result = df_clean.copy()
    result["CLUSTER"] = labels
    result.to_csv(path, index=False)
    print(f"[save_results] Labelled dataset saved → {path}")


def save_model(obj: object, path: str) -> None:
    """Pickle any sklearn-compatible object.

    Parameters
    ----------
    obj : object
        Model or scaler to persist (e.g. GaussianMixture, PCA, MinMaxScaler).
    path : str
        Target file path (e.g. ``outputs/gmm_model.pkl``).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[save_model] Object saved → {path}")


def load_model(path: str) -> object:
    """Load a pickled model or scaler.

    Parameters
    ----------
    path : str
        Path to the ``.pkl`` file.

    Returns
    -------
    object
        Unpickled object.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[load_model] Loaded from {path}")
    return obj


# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------

def set_plot_style(style: str = "whitegrid", palette: str = "deep", font_scale: float = 1.1) -> None:
    """Apply consistent Seaborn / Matplotlib aesthetics across all plots.

    Parameters
    ----------
    style : str
        Seaborn style (default ``"whitegrid"``).
    palette : str
        Seaborn palette (default ``"deep"``).
    font_scale : float
        Global font scaling factor (default 1.1).
    """
    sns.set_theme(style=style, palette=palette, font_scale=font_scale)
    plt.rcParams.update({
        "figure.dpi"       : 120,
        "savefig.dpi"      : 150,
        "savefig.bbox"     : "tight",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section divider to stdout.

    Parameters
    ----------
    title : str
        Section title text.
    width : int
        Total line width (default 60).
    """
    bar = "=" * width
    pad = (width - len(title) - 2) // 2
    print(f"\n{bar}")
    print(f"{'=' * pad} {title} {'=' * pad}")
    print(f"{bar}")
