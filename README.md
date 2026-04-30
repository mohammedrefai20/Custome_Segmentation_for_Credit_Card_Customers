# Credit Card Customer Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Unsupervised-green" />
  <img src="https://img.shields.io/badge/Clusters-7-red" />
  <img src="https://img.shields.io/badge/Silhouette-0.4671-yellow" />
</p>

> Segmenting 8,950 credit card customers into 7 distinct behavioural groups using **PCA** for dimensionality reduction and a **Gaussian Mixture Model** for clustering — enabling targeted marketing, smarter risk policies, and improved customer retention.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Cluster Profiles](#cluster-profiles)
- [Business Recommendations](#business-recommendations)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Tech Stack](#tech-stack)

---

## Overview

This project applies unsupervised machine learning to segment credit card customers based on their financial behaviour. The goal is to help banks and financial institutions:

- **Personalise** product offerings and marketing campaigns
- **Reduce** credit and default risk through targeted monitoring
- **Increase** revenue by identifying high-value customer segments
- **Improve** customer satisfaction through relevant service design

---

## Project Structure

```
credit-card-segmentation/
│
├── data/                          # Raw data (not committed — see Dataset section)
│   └── .gitkeep
│
├── notebooks/
│   └── Customer_Segmentation.ipynb  # End-to-end analysis notebook
│
├── src/                           # Modular source code
│   ├── __init__.py
│   ├── preprocessing.py           # Data loading, cleaning, log-transform, scaling
│   ├── dimensionality_reduction.py# PCA fitting and variance analysis
│   ├── clustering.py              # GMM clustering, silhouette evaluation, profiling
│   ├── visualization.py           # All plotting utilities (t-SNE, heatmap, etc.)
│   └── utils.py                   # I/O helpers, model persistence, plot style
│
├── outputs/                       # Model artefacts & results (generated at runtime)
│   └── figures/                   # Saved plots
│
├── main.py                        # CLI entry point for the full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle – Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) |
| Customers | 8,950 |
| Features | 17 behavioural variables |
| Data type | Financial transactional behaviour |

**Key features used:**

| Feature | Description |
|---------|-------------|
| `BALANCE` | Monthly average balance |
| `PURCHASES` | Total purchase amount |
| `INSTALLMENTS_PURCHASES` | Purchase amount in instalments |
| `CASH_ADVANCE` | Cash advance amount drawn |
| `CREDIT_LIMIT` | Credit limit on the card |
| `PAYMENTS` | Payments made by the customer |
| `PRC_FULL_PAYMENT` | Percentage of full payments made |
| `TENURE` | Months the account has been open |

> **Data not committed.** Download from Kaggle and place the CSV in `data/CC_GENERAL.csv`.

---

## Methodology

### 1 · Data Preprocessing

| Step | Detail |
|------|--------|
| Missing values | `CREDIT_LIMIT` → median imputation; `MINIMUM_PAYMENTS` → conditional (0 if PAYMENTS=0, else median) |
| Log transformation | Applied to 8 right-skewed features with offset `+0.01` to handle zeros |
| Scaling | `MinMaxScaler` normalises all features to `[0, 1]` |

### 2 · Dimensionality Reduction (PCA)

- **Components retained**: 12
- **Total explained variance**: 99.52%
- Reduces noise and computational cost before clustering

### 3 · Cluster Count Selection

- Gaussian Mixture Models tested for `n = 2 to 10` clusters
- Evaluated using the **Silhouette Score**
- Additional validation via 2-D **t-SNE** visualisation

### 4 · Final Clustering (GMM)

- Covariance type: `full`
- Optimal cluster count: **7**
- Best Silhouette Score: **0.46710**

---

## Results

| Metric | Value |
|--------|-------|
| Algorithm | Gaussian Mixture Model (GMM) |
| Optimal Clusters | **7** |
| Silhouette Score | **0.46710** |
| PCA Components | 12 (99.52% variance) |
| Customers Segmented | 8,950 |

The t-SNE visualisation confirmed clear visual separation between all 7 clusters, validating the structural integrity of the segmentation.

---

## Cluster Profiles

| # | Cluster Name | Customer Type | Key Characteristics |
|---|--------------|---------------|---------------------|
| 0 | **Heavy Cash Advance Users** | Credit-dependent / revolving balance | High balance & cash advance, low full-payment ratio |
| 1 | **Installment-Oriented Customers** | Planned spenders | High instalment purchases, no cash advances |
| 2 | **Cash + Installment Mixed Users** | Financially stretched | High cash advance + instalment, low full payments |
| 3 | **Cash Advance Only Customers** | Emergency cash users | Almost no purchases, high cash advance, low activity |
| 4 | **High Spenders (No Cash Advance)** | Premium loyal customers | Very high purchases & instalment, high credit limit |
| 5 | **Super Active All-Channel Users** | High-value, high-activity | High across all channels — purchases, cash, instalments |
| 6 | **Purchase-Only Customers** | Regular transactional users | Purchases only, no cash advance, no instalments |

---

## Business Recommendations

| Cluster | Strategic Action |
|---------|-----------------|
| 0 – Heavy Cash Advance | Implement dynamic risk scoring; offer instalment conversion programmes |
| 1 – Installment-Oriented | Promote long-term zero-interest plans; reward consistent repayment |
| 2 – Mixed Users | Enhanced monitoring; offer debt consolidation; delay credit limit increases |
| 3 – Cash Advance Only | Limit cash advance exposure; introduce lower-cost instalment alternatives |
| 4 – High Spenders | Premium card upgrade; exclusive loyalty rewards; selective limit increases |
| 5 – Super Active | VIP account management; cross-sell investment and insurance products |
| 6 – Purchase-Only | Nudge toward instalment usage; digital loyalty programme enrolment |

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-card-segmentation.git
cd credit-card-segmentation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) and place it at:

```
data/CC_GENERAL.csv
```

---

## Usage

### Option 1 — Jupyter Notebook (recommended for exploration)

```bash
jupyter notebook notebooks/Customer_Segmentation.ipynb
```

### Option 2 — CLI Pipeline

**Run the full pipeline with default settings (7 clusters):**

```bash
python main.py --data data/CC_GENERAL.csv
```

**Search for the optimal cluster count first:**

```bash
python main.py --data data/CC_GENERAL.csv --search
```

**Run with a custom cluster count and save figures:**

```bash
python main.py --data data/CC_GENERAL.csv --n_clusters 7 --figures outputs/figures/
```

**Headless mode (no figures):**

```bash
python main.py --data data/CC_GENERAL.csv --no_figures
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | *(required)* | Path to raw CSV file |
| `--n_clusters` | `7` | Number of GMM clusters |
| `--search` | `False` | Run silhouette search (n=2–10), then exit |
| `--output_dir` | `outputs/` | Directory for CSV and `.pkl` artefacts |
| `--figures` | `outputs/figures/` | Directory for saved plots |
| `--no_figures` | `False` | Disable all figure generation |

### Saved Artefacts

After a full run, the following files are saved to `outputs/`:

```
outputs/
├── segmented_customers.csv   # Original data + CLUSTER column
├── gmm_model.pkl             # Fitted GaussianMixture object
├── pca_model.pkl             # Fitted PCA object
├── scaler.pkl                # Fitted MinMaxScaler
└── figures/
    ├── feature_distributions.png
    ├── pca_explained_variance.png
    ├── silhouette_scores.png
    ├── cluster_distribution.png
    ├── cluster_heatmap.png
    └── tsne_clusters.png
```

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Numerical computing |
| `pandas` | ≥2.0 | Data manipulation |
| `scikit-learn` | ≥1.3 | PCA, GMM, silhouette scoring |
| `matplotlib` | ≥3.7 | Base plotting |
| `seaborn` | ≥0.13 | Statistical visualisation |
| `jupyter` | ≥7.0 | Interactive notebook environment |

---

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">Made with Python · scikit-learn · Jupyter</p>
