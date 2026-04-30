"""
src
---
Credit Card Customer Segmentation – source package.

Modules
-------
preprocessing           : data loading, cleaning, scaling
dimensionality_reduction: PCA wrapper and variance analysis
clustering              : GMM fitting, cluster selection, profiling
visualization           : all plotting utilities
utils                   : I/O helpers, style configuration
"""

from .preprocessing import preprocess_pipeline, load_data, handle_missing_values
from .dimensionality_reduction import fit_pca, plot_explained_variance
from .clustering import fit_gmm, find_optimal_clusters, print_cluster_profiles
from .visualization import plot_tsne, plot_cluster_heatmap, plot_cluster_distribution
from .utils import save_results, save_model, set_plot_style

__all__ = [
    "preprocess_pipeline",
    "load_data",
    "handle_missing_values",
    "fit_pca",
    "plot_explained_variance",
    "fit_gmm",
    "find_optimal_clusters",
    "print_cluster_profiles",
    "plot_tsne",
    "plot_cluster_heatmap",
    "plot_cluster_distribution",
    "save_results",
    "save_model",
    "set_plot_style",
]
