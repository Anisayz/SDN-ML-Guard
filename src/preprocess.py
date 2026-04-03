"""
preprocess.py — Offline preprocessing for CIC-IDS2018
======================================================

Outputs (all under data/processed/):
    train.pkl, test.pkl        — stratified 80/20 split (features + labels)
    benign_train.pkl           — benign-only rows from train set (for Isolation Forest)
    scaler.pkl                 — fitted StandardScaler
    label_encoder.pkl          — fitted LabelEncoder
    feature_list.json          — ordered feature names used at runtime
"""

import json
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (edit RAW_DIR if your layout differs)
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent.parent
RAW_DIR      = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (derived from EDA)
# ---------------------------------------------------------------------------
LABEL_COL    = "Label"
BENIGN_LABEL = "Benign"
BENIGN_CAP   = 100_000
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Columns present only in Thuesday-20-02 — not features, drop if found
IDENTITY_COLS = ["Flow ID", "Src IP", "Dst IP", "Src Port"]

# Timestamp is a string datetime — not a feature
DROP_COLS = ["Timestamp"]

# All-zero bulk columns confirmed in EDA (unique=3, values=[0,'0','col_name'])
ZERO_BULK_COLS = [
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
]

# Rogue label value — header rows repeated mid-CSV (CIC-IDS2018 artifact)
ROGUE_LABEL = "Label"


# ---------------------------------------------------------------------------
# Step 1 — Load & sample
# ---------------------------------------------------------------------------
def load_and_sample(raw_dir: Path) -> pd.DataFrame:
    """
    For each CSV:
      - Sample up to BENIGN_CAP benign rows  (random, reproducible)
      - Keep ALL attack rows
      - Drop identity / timestamp columns on the fly to save RAM
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for f in csv_files:
        log.info(f"Loading {f.name} ...")
        df = pd.read_csv(f, low_memory=False)

        # Strip whitespace from column names (CIC quirk)
        df.columns = df.columns.str.strip()

        # Strip whitespace from label values
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

        # Drop identity + timestamp columns immediately (save RAM)
        cols_to_drop = [c for c in IDENTITY_COLS + DROP_COLS if c in df.columns]
        if cols_to_drop:
            log.info(f"  Dropping columns: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)

        # Remove rogue header rows (Label == 'Label')
        rogue_mask = df[LABEL_COL] == ROGUE_LABEL
        if rogue_mask.any():
            log.info(f"  Removing {rogue_mask.sum()} rogue header rows")
            df = df[~rogue_mask]

        # Sample
        benign_mask = df[LABEL_COL] == BENIGN_LABEL
        benign_df   = df[benign_mask].sample(
            n=min(BENIGN_CAP, benign_mask.sum()), random_state=RANDOM_STATE
        )
        attack_df = df[~benign_mask]

        log.info(
            f"  → {len(benign_df):>7,} benign  +  {len(attack_df):>7,} attack"
        )
        frames.append(pd.concat([benign_df, attack_df], ignore_index=True))

    df_all = pd.concat(frames, ignore_index=True)
    log.info(f"Combined shape after sampling: {df_all.shape}")
    return df_all


# ---------------------------------------------------------------------------
# Step 2 — Clean
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix all data-quality issues found in EDA:
      1. Force every feature column to numeric (kills leftover string header echoes)
      2. Fill NaN in Flow Byts/s with 0 (zero-duration flows)
      3. Drop the 6 all-zero bulk columns
      4. Replace any remaining NaN / Inf with 0
    """
    log.info("Cleaning ...")

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    # 1. Force numeric — 'FIN Flag Cnt' strings → NaN, then we handle below
    log.info("  Casting all feature columns to numeric ...")
    df[feature_cols] = df[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # 2. Flow Byts/s NaN → 0  (confirmed: zero-duration flows, 0.1% of data)
    if "Flow Byts/s" in df.columns:
        n_nan = df["Flow Byts/s"].isna().sum()
        if n_nan:
            log.info(f"  Flow Byts/s: filling {n_nan:,} NaN with 0")
            df["Flow Byts/s"] = df["Flow Byts/s"].fillna(0)

    # 3. Drop all-zero bulk columns
    bulk_to_drop = [c for c in ZERO_BULK_COLS if c in df.columns]
    if bulk_to_drop:
        log.info(f"  Dropping zero-bulk columns: {bulk_to_drop}")
        df.drop(columns=bulk_to_drop, inplace=True)

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    
    # 4. Catch any remaining NaN / Inf (safety net)
    remaining_nan = df[feature_cols].isna().sum().sum()
    remaining_inf = np.isinf(
        df[[c for c in feature_cols if c in df.columns]].select_dtypes(include=[np.number])
    ).sum().sum()

    if remaining_nan or remaining_inf:
        log.warning(
            f"  Safety fill: {remaining_nan:,} NaN  +  {remaining_inf:,} Inf → 0"
        )
        feature_cols_now = [c for c in df.columns if c != LABEL_COL]
        df[feature_cols_now] = (
            df[feature_cols_now]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
    
    post_cast_const = [c for c in df.columns if c != LABEL_COL and df[c].nunique() <= 1]
    if post_cast_const:
       log.warning(f"Constant columns after casting: {post_cast_const}")
 
    log.info(f"  Shape after cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Encode labels
# ---------------------------------------------------------------------------
def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Fit a LabelEncoder on the Label column.
    Returns df with an extra 'label_enc' integer column and the fitted encoder.
    """
    log.info("Encoding labels ...")
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df[LABEL_COL])

    log.info("  Classes:")
    for i, cls in enumerate(le.classes_):
        count = (df["label_enc"] == i).sum()
        log.info(f"    [{i:2d}] {cls:40s}  {count:>8,}")

    return df, le


# ---------------------------------------------------------------------------
# Step 4 — Train / test split
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 split on the encoded label so every class is represented
    in both sets proportionally.
    """
    log.info(f"Splitting 80/20 (stratified) ...")
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_enc"],
    )
    log.info(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5 — Scale features
# ---------------------------------------------------------------------------
def scale(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on training features only, transform both splits.
    Scaler is fitted AFTER the split so test data never leaks into the scaler.
    """
    log.info("Fitting StandardScaler on training data ...")
    scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    log.info(f"  Scaled {len(feature_cols)} features")
    return train_df, test_df, scaler


# ---------------------------------------------------------------------------
# Step 6 — Save artefacts
# ---------------------------------------------------------------------------
def save_artifacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_df_raw: pd.DataFrame,
    benign_train_df: pd.DataFrame,
    scaler: StandardScaler,
    benign_scaler: StandardScaler,
    le: LabelEncoder,
    feature_cols: list[str],
    out_dir: Path,
) -> None:
    log.info(f"Saving artefacts to {out_dir} ...")

    # DataFrames
    train_df.to_pickle(out_dir / "train.pkl")
    test_df.to_pickle(out_dir / "test.pkl")
    benign_train_df.to_pickle(out_dir / "benign_train.pkl")
    test_df_raw.to_pickle(out_dir / "test_raw.pkl")   # NEW
    log.info("  Saved test_raw.pkl")
    log.info("  Saved train.pkl, test.pkl, benign_train.pkl")

    # Scaler
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info("  Saved scaler.pkl")

    # Benign-only scaler (for Isolation Forest)
    with open(out_dir / "benign_scaler.pkl", "wb") as f:
        pickle.dump(benign_scaler, f)
    log.info("  Saved benign_scaler.pkl")

    # Label encoder
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    log.info("  Saved label_encoder.pkl")

    # Feature list — critical for runtime predict.py
    feature_list_path = out_dir.parent / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info(f"  Saved feature_list.json  ({len(feature_cols)} features)")

    # Quick sanity summary
    summary = {
        "train_rows"       : len(train_df),
        "test_rows"        : len(test_df),
        "benign_train_rows": len(benign_train_df),
        "n_features"       : len(feature_cols),
        "n_classes"        : len(le.classes_),
        "classes"          : le.classes_.tolist(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  Saved summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("CIC-IDS2018 Preprocessing Pipeline")
    log.info("=" * 60)

    # 1. Load & sample
    df = load_and_sample(RAW_DIR)

    # 2. Clean
    df = clean(df)

    # 3. Encode labels
    df, le = encode_labels(df)

    # 4. Determine final feature columns (everything except label columns)
    feature_cols = [c for c in df.columns if c not in (LABEL_COL, "label_enc")]
    log.info(f"Final feature count: {len(feature_cols)}")

    # 5. Train / test split
    train_df, test_df = split(df)
    test_df_raw = test_df.copy()          # snapshot before scaling
    # 6. Isolate benign-only training rows BEFORE scaling
    #    (Isolation Forest needs the same scaled space, so we do this after split
    #     but we'll re-extract from the scaled train below)
    benign_mask_train = train_df[LABEL_COL] == BENIGN_LABEL

    #6.1 fit a benign-only scaler on raw unscaled benign rows
    log.info("Fitting benign-only StandardScaler for Isolation Forest ...")
    benign_raw = train_df[benign_mask_train][feature_cols].reset_index(drop=True)
    benign_scaler = StandardScaler()
    benign_scaled = benign_scaler.fit_transform(benign_raw)
    benign_train_df = train_df[benign_mask_train].reset_index(drop=True).copy()
    benign_train_df[feature_cols] = benign_scaled
    log.info(f"  Benign-only scaler fitted on {len(benign_train_df):,} rows")

    # 7. Scale (fit on train only)
    train_df, test_df, scaler = scale(train_df, test_df, feature_cols)

    

    # 9. Save everything
    save_artifacts(
        train_df, test_df, test_df_raw, benign_train_df,
        scaler, benign_scaler, le, feature_cols,
        PROCESSED_DIR,
    )

    log.info("=" * 60)
    log.info("Preprocessing complete.")
    log.info(f"  Processed files → {PROCESSED_DIR}")
    log.info(f"  Feature list    → {PROCESSED_DIR.parent / 'feature_list.json'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()