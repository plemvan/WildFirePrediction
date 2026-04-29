"""
Data loading and preparation for the WildFire dataset.

This module is the single entry point for data access.
It loads the raw CSV from S3 and returns a clean DataFrame
ready for model training.

Usage
-----
    from src.data.df_aggregated import load_data_from_s3

    df = load_data_from_s3()
    # df has columns: pr, rmax, rmin, sph, srad, tmmn, tmmx,
    #                 vs, vpd, fm100, fm1000, erc, bi, etr, pet, label
"""

import ast
import os
import zipfile

import pandas as pd
import s3fs
from dotenv import load_dotenv


load_dotenv()

FEATURES = [
    "pr", "rmax", "rmin", "sph", "srad",
    "tmmn", "tmmx", "vs", "vpd",
    "fm100", "fm1000", "erc", "bi", "etr", "pet",
]

TARGET_COLUMN = "label"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw CSV into a model-ready DataFrame.

    Steps
    -----
    1. Parse the ``Wildfire`` column (stored as a Python tuple string)
       into a binary ``label`` column (1 = fire, 0 = no fire).
    2. Keep only the 15 meteorological features and ``label``.
    3. Cast all columns to float / int and drop rows with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as loaded from S3.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns FEATURES + [label].
    """
    # Parse "Wildfire": e.g. "(1, 'Yes')" -> 1, "(0, 'No')" -> 0
    df[TARGET_COLUMN] = df["Wildfire"].apply(
        lambda x: 1 if ast.literal_eval(x)[1].lower() == "yes" else 0
    )

    # Keep only relevant columns
    cols = FEATURES + [TARGET_COLUMN]
    df = df[cols].copy()

    # Ensure numeric types (some values may have commas or extra spaces)
    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".").str.strip(),
            errors="coerce",
        )

    df = df.dropna()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df.reset_index(drop=True)


def load_data_from_s3() -> pd.DataFrame:
    """
    Load and clean the wildfire dataset from S3 / MinIO.

    Reads ``S3_ENDPOINT_URL`` and ``S3_BUCKET`` from environment variables
    (set in your ``.env`` file or shell).

    If ``ACCESS_KEY`` is not set, connects anonymously (public bucket).

    The raw file ``df_aggregated.parquet`` is fetched from S3, with automatic
    fallback to ``df_aggregated.csv.zip`` if the parquet file is not found.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with columns:
        pr, rmax, rmin, sph, srad, tmmn, tmmx, vs, vpd,
        fm100, fm1000, erc, bi, etr, pet, label.

    Raises
    ------
    EnvironmentError
        If ``S3_ENDPOINT_URL`` or ``S3_BUCKET`` are not set.
    """
    endpoint = os.getenv("S3_ENDPOINT_URL")
    bucket = os.getenv("S3_BUCKET")

    if not endpoint or not bucket:
        raise EnvironmentError(
            "S3_ENDPOINT_URL and S3_BUCKET must be set in the environment or .env file."
        )

    access_key = os.getenv("ACCESS_KEY") or None
    secret_key = os.getenv("SECRET_KEY") or None
    session_token = os.getenv("SESSION_TOKEN") or None

    # Use anonymous access if no credentials are provided (public bucket)
    anon = not bool(access_key)

    fs = s3fs.S3FileSystem(
        anon=anon,
        key=access_key,
        secret=secret_key,
        token=session_token,
        client_kwargs={
            "endpoint_url": endpoint,
            "verify": False,
        },
    )

    try:
        file_path = f"{bucket}/df_aggregated.parquet"
        with fs.open(file_path, "rb") as f:
            df = pd.read_parquet(f)

    except Exception as e:
        print(f"Try to read parquet is non successfull : {e}")
        file_path = f"{bucket}/df_aggregated.csv.zip"
        with fs.open(file_path, "rb") as f:
            with zipfile.ZipFile(f) as z:
                csv_files = [
                    name for name in z.namelist()
                    if name.endswith(".csv") and not name.startswith("__MACOSX")
                ]
                df = pd.read_csv(z.open(csv_files[0]))

    return _clean(df)