"""
main.py
-------
End-to-end pipeline entry point for the
Credit Card Customer Segmentation project.

Usage
-----
    python main.py --data data/CC_GENERAL.csv

Arguments
---------
    --data       Path to the raw CSV file            (required)
    --n_clusters Number of GMM clusters              (default: 7)
    --search     Run silhouette search (2-10 clusters) and then exit
    --output_dir Directory for CSV / pickle outputs  (default: outputs/)
    --figures    Directory for saved figures          (default: outputs/figures/)
    --no_figures Skip figure generation (useful in CI / headless mode)
"""

import argparse
import os
import sys

from src.utils import set_plot_style, print_section, save_results, save_model
from src.preprocessing import (
    preprocess_pipeline, apply_log_transform,
    handle_missing_values, load_data, LOG_TRANSFORM_FEATURES,
)
from src.dimensionality_reduction import fit_pca, plot_explained_variance
from src.clustering import (
    find_optimal_clusters, fit_gmm, build_cluster_summary,
    print_cluster_profiles, plot_silhouette_scores, CLUSTER_PROFILES,
)
from src.visualization import (
    plot_tsne, plot_cluster_heatmap,
    plot_cluster_distribution, plot_feature_distributions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Credit Card Customer Segmentation -- GMM + PCA pipeline"
    )
    parser.add_argument("--data",       required=True)
    parser.add_argument("--n_clusters", type=int, default=7)
    parser.add_argument("--search",     action="store_true")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--figures",    default="outputs/figures")
    parser.add_argument("--no_figures", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_plot_style()

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_figures:
        os.makedirs(args.figures, exist_ok=True)

    # 1. Preprocessing
    print_section("1 . PREPROCESSING")
    X_scaled, df_clean, scaler = preprocess_pipeline(args.data)

    if not args.no_figures:
        df_raw = load_data(args.data)
        df_imp = handle_missing_values(df_raw)
        df_log = apply_log_transform(df_imp)
        feats  = [f for f in LOG_TRANSFORM_FEATURES if f in df_imp.columns]
        plot_feature_distributions(
            df_before=df_imp, df_after=df_log, features=feats[:6],
            save_path=os.path.join(args.figures, "feature_distributions.png"),
        )

    # 2. PCA
    print_section("2 . PCA")
    X_pca, pca = fit_pca(X_scaled)
    if not args.no_figures:
        plot_explained_variance(
            pca, save_path=os.path.join(args.figures, "pca_explained_variance.png"),
        )

    # 3a. Optional cluster count search
    if args.search:
        print_section("3 . CLUSTER COUNT SEARCH")
        scores = find_optimal_clusters(X_pca)
        if not args.no_figures:
            plot_silhouette_scores(
                scores, save_path=os.path.join(args.figures, "silhouette_scores.png"),
            )
        print("\nSearch complete. Re-run without --search to fit the final model.")
        sys.exit(0)

    # 3b. Fit GMM
    print_section(f"3 . GMM CLUSTERING  (n={args.n_clusters})")
    labels, gmm = fit_gmm(X_pca, n_components=args.n_clusters)

    # 4. Profiles
    print_section("4 . CLUSTER PROFILES")
    print_cluster_profiles(labels)
    cluster_names = {k: v["name"] for k, v in CLUSTER_PROFILES.items()}
    summary = build_cluster_summary(df_clean, labels)
    print("\nCluster mean statistics:")
    print(summary.to_string())

    # 5. Visualise
    if not args.no_figures:
        print_section("5 . VISUALISATION")
        plot_cluster_distribution(
            labels, cluster_names=cluster_names,
            save_path=os.path.join(args.figures, "cluster_distribution.png"),
        )
        plot_cluster_heatmap(
            summary, save_path=os.path.join(args.figures, "cluster_heatmap.png"),
        )
        plot_tsne(
            X_pca, labels, cluster_names=cluster_names,
            save_path=os.path.join(args.figures, "tsne_clusters.png"),
        )

    # 6. Save artefacts
    print_section("6 . SAVING ARTEFACTS")
    save_results(df_clean, labels, path=os.path.join(args.output_dir, "segmented_customers.csv"))
    save_model(gmm,    os.path.join(args.output_dir, "gmm_model.pkl"))
    save_model(pca,    os.path.join(args.output_dir, "pca_model.pkl"))
    save_model(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    print_section("DONE")


if __name__ == "__main__":
    main()
