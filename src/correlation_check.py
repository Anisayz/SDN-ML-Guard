"""
correlation_check.py — Post-preprocessing correlation analysis
==============================================================
Run AFTER preprocess.py has produced data/processed/train.pkl

    python src/correlation_check.py

Outputs (all under data/):
    corr_report.json          — full list of high-corr pairs + drop recommendations
    corr_heatmap.png          — heatmap of features that have at least one high-corr pair
    feature_list_pruned.json  — updated feature list with redundant columns removed
                                (drop this in place of feature_list.json before training)

Threshold: |r| >= 0.95  (Pearson on StandardScaler-normalized features)
    — adjust CORR_THRESHOLD below if you want to be more / less aggressive
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_DIR      = BASE_DIR / "data"

TRAIN_PKL          = PROCESSED_DIR / "train.pkl"
FEATURE_LIST_JSON  = DATA_DIR / "feature_list.json"
OUT_REPORT         = DATA_DIR / "corr_report.json"
OUT_HEATMAP        = DATA_DIR / "corr_heatmap.png"
OUT_PRUNED_FEATURES = DATA_DIR / "feature_list_pruned.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CORR_THRESHOLD = 0.95   # |r| >= this → flagged as redundant pair
LABEL_COLS     = ["Label", "label_enc"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_features(train_pkl: Path, feature_list_json: Path) -> pd.DataFrame:
    log.info(f"Loading {train_pkl.name} ...")
    train_df = pd.read_pickle(train_pkl)

    with open(feature_list_json) as f:
        feature_cols = json.load(f)

    # Keep only feature columns that actually exist (safety check)
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    log.info(f"  {len(train_df):,} rows  ×  {len(feature_cols)} features")

    return train_df[feature_cols]


def compute_corr_matrix(X: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing Pearson correlation matrix (this may take ~1 min) ...")
    corr = X.corr(method="pearson")
    log.info("  Done.")
    return corr


def find_high_corr_pairs(corr: pd.DataFrame, threshold: float) -> list[dict]:
    """
    Return a list of dicts, one per high-corr pair:
        { feat_a, feat_b, correlation, drop_candidate }
    drop_candidate = the column with the lower mean absolute correlation
    to the rest of the dataset (i.e. the less "connected" one — safer to drop).
    """
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []

    # Mean absolute correlation for each feature (proxy for "importance to others")
    mean_abs_corr = corr.abs().mean()

    for col in upper.columns:
        high = upper[col][upper[col].abs() >= threshold]
        for row_feat, r_val in high.items():
            # Drop whichever is less correlated with everything else
            if mean_abs_corr[col] <= mean_abs_corr[row_feat]:
                drop_candidate = col
            else:
                drop_candidate = row_feat

            pairs.append({
                "feat_a"       : row_feat,
                "feat_b"       : col,
                "correlation"  : round(float(r_val), 6),
                "drop_candidate": drop_candidate,
            })

    # Sort by absolute correlation descending
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs


def pick_cols_to_drop(pairs: list[dict]) -> list[str]:
    """
    Greedy deduplication: iterate pairs by |r| descending.
    Once a column is marked for drop, don't re-evaluate it.
    This avoids cascading drops where A→drop B, B→drop C,
    but A and C are actually fine together.
    """
    dropped = set()
    for p in pairs:
        a, b = p["feat_a"], p["feat_b"]
        if a in dropped or b in dropped:
            continue          # one side already gone — pair resolved
        dropped.add(p["drop_candidate"])
    return sorted(dropped)


def plot_heatmap(corr: pd.DataFrame, cols_involved: list[str], out_path: Path) -> None:
    """
    Plot a heatmap of only the features involved in at least one high-corr pair.
    Full 72×72 heatmap is unreadable; this focused view is actionable.
    """
    if not cols_involved:
        log.info("  No high-corr pairs → skipping heatmap.")
        return

    sub = corr.loc[cols_involved, cols_involved]
    n   = len(cols_involved)
    fig_size = max(12, n * 0.55)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        sub,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        annot=(n <= 30),          # only annotate if small enough to read
        fmt=".2f",
        linewidths=0.3,
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(
        f"High-correlation feature pairs  (|r| ≥ {CORR_THRESHOLD})\n"
        f"{n} features involved",
        fontsize=13, pad=14,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Heatmap saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("Post-preprocessing Correlation Check")
    log.info("=" * 60)

    # 1. Load
    X = load_features(TRAIN_PKL, FEATURE_LIST_JSON)

    # 2. Correlation matrix
    corr = compute_corr_matrix(X)

    # 3. Find high-corr pairs
    pairs = find_high_corr_pairs(corr, CORR_THRESHOLD)
    log.info(f"High-corr pairs found (|r| ≥ {CORR_THRESHOLD}): {len(pairs)}")

    if pairs:
        log.info("  Top 20 pairs:")
        for p in pairs[:20]:
            log.info(
                f"    {p['feat_a']:35s}  ↔  {p['feat_b']:35s}"
                f"  r={p['correlation']:+.4f}  → drop: {p['drop_candidate']}"
            )

    # 4. Decide what to drop (greedy dedup)
    cols_to_drop = pick_cols_to_drop(pairs)
    log.info(f"\nColumns recommended for removal: {len(cols_to_drop)}")
    for c in cols_to_drop:
        log.info(f"  - {c}")

    # 5. Build pruned feature list
    with open(FEATURE_LIST_JSON) as f:
        original_features = json.load(f)

    pruned_features = [c for c in original_features if c not in cols_to_drop]
    log.info(
        f"\nFeature count:  {len(original_features)} original  →  "
        f"{len(pruned_features)} after pruning  "
        f"({len(cols_to_drop)} removed)"
    )

    # 6. Save heatmap
    cols_involved = sorted(
        set(p["feat_a"] for p in pairs) | set(p["feat_b"] for p in pairs)
    )
    plot_heatmap(corr, cols_involved, OUT_HEATMAP)

    # 7. Save report JSON
    report = {
        "threshold"            : CORR_THRESHOLD,
        "n_original_features"  : len(original_features),
        "n_pruned_features"    : len(pruned_features),
        "n_pairs_found"        : len(pairs),
        "cols_to_drop"         : cols_to_drop,
        "pruned_feature_list"  : pruned_features,
        "all_pairs"            : pairs,
    }
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → {OUT_REPORT}")

    # 8. Save pruned feature list (use this instead of feature_list.json at training time)
    with open(OUT_PRUNED_FEATURES, "w") as f:
        json.dump(pruned_features, f, indent=2)
    log.info(f"Pruned feature list saved → {OUT_PRUNED_FEATURES}")

    # 9. Final recommendation
    log.info("=" * 60)
    if not cols_to_drop:
        log.info("✅  No redundant features found. Use feature_list.json as-is.")
    else:
        log.info(
            f"⚠️   {len(cols_to_drop)} redundant features flagged.\n"
            f"     Review corr_report.json + corr_heatmap.png,\n"
            f"     then use feature_list_pruned.json for training\n"
            f"     (or keep originals — RF is robust to correlated features)."
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()