"""
preprocessing.py
----------------
Data loading, cleaning, and feature engineering for the
Credit Card Customer Segmentation project.

Steps
-----
1. Load raw CSV data
2. Handle missing values (CREDIT_LIMIT, MINIMUM_PAYMENTS)
3. Apply log transformation to right-skewed features
4. Scale features with MinMaxScaler
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Features that exhibit strong right skew and benefit from log transformation
LOG_TRANSFORM_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
]

LOG_OFFSET = 0.01  # added before log to avoid log(0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw credit-card dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g. ``data/CC GENERAL.csv``).

    Returns
    -------
    pd.DataFrame
        Raw dataframe with the CUST_ID column dropped (not needed for
        modelling).
    """
    df = pd.read_csv(filepath)
    if "CUST_ID" in df.columns:
        df = df.drop(columns=["CUST_ID"])
    print(f"[load_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using domain-aware strategies.

    Strategy
    --------
    * **CREDIT_LIMIT** – filled with the column median (robust to outliers).
    * **MINIMUM_PAYMENTS** – two-step:
        - If ``PAYMENTS == 0``, set ``MINIMUM_PAYMENTS = 0``.
        - Otherwise, fill remaining NaNs with the column median.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (output of :func:`load_data`).

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values resolved.
    """
    df = df.copy()

    # CREDIT_LIMIT – median imputation
    credit_limit_median = df["CREDIT_LIMIT"].median()
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(credit_limit_median)
    print(
        f"[handle_missing_values] CREDIT_LIMIT NaNs filled with median "
        f"({credit_limit_median:.2f})."
    )

    # MINIMUM_PAYMENTS – conditional imputation
    min_pay_median = df["MINIMUM_PAYMENTS"].median()
    df.loc[df["PAYMENTS"] == 0, "MINIMUM_PAYMENTS"] = 0
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(min_pay_median)
    print(
        f"[handle_missing_values] MINIMUM_PAYMENTS NaNs filled conditionally "
        f"(median = {min_pay_median:.2f})."
    )

    remaining_na = df.isnull().sum().sum()
    print(f"[handle_missing_values] Remaining NaN count: {remaining_na}")
    return df


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p-like transformation to right-skewed numerical features.

    The transformation used is ``log(x + LOG_OFFSET)`` where
    ``LOG_OFFSET = 0.01`` prevents ``log(0)`` errors for zero-valued entries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after missing-value imputation.

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed columns (in-place replacement).
    """
    df = df.copy()
    cols_to_transform = [c for c in LOG_TRANSFORM_FEATURES if c in df.columns]
    for col in cols_to_transform:
        df[col] = np.log(df[col] + LOG_OFFSET)
    print(
        f"[apply_log_transform] Log-transformed {len(cols_to_transform)} "
        f"features: {cols_to_transform}"
    )
    return df


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler]:
    """Normalise all features to [0, 1] using MinMaxScaler.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with log-transformed features.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix.
    scaler : MinMaxScaler
        Fitted scaler (keep this for inverse-transforming predictions later).
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df)
    print(
        f"[scale_features] MinMaxScaler applied → "
        f"shape {X_scaled.shape}, range [{X_scaled.min():.2f}, {X_scaled.max():.2f}]."
    )
    return X_scaled, scaler


def preprocess_pipeline(filepath: str) -> tuple[np.ndarray, pd.DataFrame, MinMaxScaler]:
    """End-to-end preprocessing pipeline.

    Convenience wrapper that chains :func:`load_data`,
    :func:`handle_missing_values`, :func:`apply_log_transform`, and
    :func:`scale_features`.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix ready for PCA.
    df_clean : pd.DataFrame
        Cleaned (but un-scaled) dataframe – useful for cluster profiling.
    scaler : MinMaxScaler
        Fitted scaler.
    """
    df_raw = load_data(filepath)
    df_clean = handle_missing_values(df_raw)
    df_log = apply_log_transform(df_clean)
    X_scaled, scaler = scale_features(df_log)
    return X_scaled, df_clean, scaler
