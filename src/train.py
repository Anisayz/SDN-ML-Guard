"""
train.py — Train Random Forest (classifier) + Autoencoder (anomaly detector)
=============================================================================

Reads from  data/processed/  → writes models to  models/

Outputs:
    models/classifier.pkl   — RandomForestClassifier (multi-class)
    models/anomaly.pkl      — AutoencoderWrapper (PyTorch, benign-only)
"""

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from models import Autoencoder, AutoencoderWrapper
from torch.utils.data import DataLoader, TensorDataset

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
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# --- Random Forest ---
# min_samples_leaf=2: finer boundaries for overlapping classes.
# max_features='log2': more features per split — helps Dst Port disambiguate
#   FTP-BruteForce from DoS-SlowHTTPTest.
# MAX_CLASS_WEIGHT=20: prevents hallucination of rare labels (was 50x).
MAX_CLASS_WEIGHT = 20.0

RF_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = None,
    min_samples_leaf = 2,
    max_features     = "log2",
    class_weight     = None,    # set dynamically in train_classifier()
    oob_score        = True,
    n_jobs           = -1,
    random_state     = 42,
    verbose          = 1,
)

# --- Autoencoder ---
# Architecture: 72 → 32 → 16 → 32 → 72
# Trained on benign-only flows scaled with benign_scaler.
# Anomaly score at runtime = reconstruction error (MSE per sample).
# Higher error = more anomalous.
AE_PARAMS = dict(
    hidden_dims  = [32, 16],   # encoder dims; decoder mirrors these
    epochs       = 50,
    batch_size   = 1024,
    lr           = 1e-3,
    weight_decay = 1e-5,       # L2 regularisation — prevents memorising noise
    patience     = 5,          # early stopping — stop if val loss doesn't improve
    val_split    = 0.1,        # fraction of benign_train held out for validation
)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], object]:
    log.info("Loading processed data ...")

    train_df     = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    benign_df    = pd.read_pickle(PROCESSED_DIR / "benign_train.pkl")
    feature_list = json.loads((BASE_DIR / "data" / "feature_list.json").read_text())

    X_train  = train_df[feature_list].values.astype(np.float32)
    y_train  = train_df["label_enc"].values.astype(np.int32)
    X_benign = benign_df[feature_list].values.astype(np.float32)

    log.info(f"  X_train  shape : {X_train.shape}")
    log.info(f"  y_train  shape : {y_train.shape}")
    log.info(f"  X_benign shape : {X_benign.shape}")

    le_path = PROCESSED_DIR / "label_encoder.pkl"
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    unique, counts = np.unique(y_train, return_counts=True)
    log.info("  Training class distribution:")
    for cls_id, cnt in zip(unique, counts):
        log.info(f"    [{cls_id:2d}] {le.classes_[cls_id]:40s}  {cnt:>8,}")

    return X_train, y_train, X_benign, feature_list, le


def save_model(model, path: Path, name: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / 1e6
    log.info(f"  Saved {name} → {path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Phase 5a — Random Forest
# ---------------------------------------------------------------------------
def train_classifier(
    X_train: np.ndarray, y_train: np.ndarray, le
) -> RandomForestClassifier:
    log.info("=" * 55)
    log.info("Phase 5a — Training Random Forest Classifier")
    log.info("=" * 55)
    log.info(f"  Params: {RF_PARAMS}")

    # Capped class weights
    raw_weights    = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    capped_weights = np.minimum(raw_weights, MAX_CLASS_WEIGHT)
    weight_dict    = dict(enumerate(capped_weights))

    log.info(f"  Class weights (capped at {MAX_CLASS_WEIGHT}x):")
    for cls_id, w in weight_dict.items():
        log.info(f"    [{cls_id:2d}]  raw={raw_weights[cls_id]:7.2f}  capped={w:6.2f}")

    # SMOTE on Infilteration
    infilteration_id = list(le.classes_).index("Infilteration")
    log.info(f"  Applying SMOTE to Infilteration (class {infilteration_id}) ...")
    sm = SMOTE(
        sampling_strategy={infilteration_id: 300_000},
        random_state=42,
        k_neighbors=5,
    )
    X_train, y_train = sm.fit_resample(X_train, y_train)
    log.info(f"  After SMOTE: {X_train.shape[0]:,} rows")

    # Recompute weights after SMOTE (Infilteration distribution changed)
    raw_weights    = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    capped_weights = np.minimum(raw_weights, MAX_CLASS_WEIGHT)
    weight_dict    = dict(enumerate(capped_weights))
    params         = {**RF_PARAMS, "class_weight": weight_dict}

    t0  = time.time()
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    log.info(f"  Training time : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    log.info(f"  OOB accuracy  : {clf.oob_score_:.4f}")

    fi = pd.Series(
        clf.feature_importances_,
        index=json.loads((BASE_DIR / "data" / "feature_list.json").read_text()),
    ).sort_values(ascending=False)
    log.info("  Top-10 feature importances:")
    for feat, imp in fi.head(10).items():
        bar = "█" * int(imp * 200)
        log.info(f"    {feat:35s}  {imp:.4f}  {bar}")

    return clf


# ---------------------------------------------------------------------------
# Phase 5b — Autoencoder anomaly detector
# ---------------------------------------------------------------------------
def train_anomaly_detector(X_benign: np.ndarray, clf, feature_list: list[str]) -> AutoencoderWrapper:
    log.info("=" * 55)
    log.info("Phase 5b — Training Autoencoder (benign-only)")
    log.info("=" * 55)
    log.info(f"  Training on {X_benign.shape[0]:,} benign flows")
    log.info(f"  Params: {AE_PARAMS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  Device: {device}")

    input_dim = X_benign.shape[1]

    # Train / validation split
    n_val    = int(len(X_benign) * AE_PARAMS["val_split"])
    n_train  = len(X_benign) - n_val
    idx      = np.random.RandomState(42).permutation(len(X_benign))
    X_tr     = X_benign[idx[:n_train]]
    X_val    = X_benign[idx[n_train:]]
    log.info(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}")

    # DataLoaders
    tr_tensor  = torch.tensor(X_tr,  dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    tr_loader  = DataLoader(
        TensorDataset(tr_tensor),
        batch_size=AE_PARAMS["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=AE_PARAMS["batch_size"] * 4,
        shuffle=False,
        num_workers=0,
    )

    # Build feature weights from RF importances
    # Features the RF considers most discriminative get higher weight —
    # the autoencoder is forced to reconstruct them precisely, so it
    # fails harder on attack flows that violate those feature patterns.
    fi_raw = clf.feature_importances_                         # shape (72,)
    # Normalise to [1, max_weight] range so all features still matter
    fi_min, fi_max = fi_raw.min(), fi_raw.max()
    AE_WEIGHT_MAX = 10.0   # most important feature gets 10x weight
    feature_weights = 1.0 + (AE_WEIGHT_MAX - 1.0) * (fi_raw - fi_min) / (fi_max - fi_min + 1e-8)
    log.info(f"  Feature weights: min={feature_weights.min():.2f}  "
             f"max={feature_weights.max():.2f}  mean={feature_weights.mean():.2f}")
    top5_idx = fi_raw.argsort()[::-1][:5]
    for idx in top5_idx:
        log.info(f"    {feature_list[idx]:35s}  importance={fi_raw[idx]:.4f}  weight={feature_weights[idx]:.2f}")

    # Model, optimiser — loss is now model.weighted_loss()
    model     = Autoencoder(input_dim, AE_PARAMS["hidden_dims"], feature_weights).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=AE_PARAMS["lr"],
        weight_decay=AE_PARAMS["weight_decay"],
    )

    log.info(f"  Architecture: {input_dim} → "
             f"{' → '.join(str(h) for h in AE_PARAMS['hidden_dims'])} → "
             f"{' → '.join(str(h) for h in reversed(AE_PARAMS['hidden_dims'][:-1]))} → "
             f"{input_dim}")
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Total parameters: {total_params:,}")

    # Training loop with early stopping
    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    t0             = time.time()

    for epoch in range(1, AE_PARAMS["epochs"] + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for (batch,) in tr_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            loss  = model.weighted_loss(batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(batch)  # weighted loss
        train_loss /= len(X_tr)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch    = batch.to(device)
                val_loss += model.weighted_loss(batch).item() * len(batch)
        val_loss /= len(X_val)

        # Log every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:3d}/{AE_PARAMS['epochs']}  "
                     f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= AE_PARAMS["patience"]:
                log.info(f"  Early stopping at epoch {epoch} "
                         f"(best val_loss={best_val_loss:.6f})")
                break

    elapsed = time.time() - t0
    log.info(f"  Training time : {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()

    # Score stats on benign training data (sanity check)
    with torch.no_grad():
        errors = model.reconstruction_error(
            torch.tensor(X_benign, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # Negate for IF-compatible convention (more negative = more anomalous)
    scores = -errors
    log.info(f"  Reconstruction error stats on benign train:")
    log.info(f"    mean={errors.mean():.6f}  std={errors.std():.6f}")
    log.info(f"    min={errors.min():.6f}   max={errors.max():.6f}")
    log.info(f"    p1={np.percentile(errors,99):.6f}  "   # p99 of error = p1 of -error
             f"p50={np.percentile(errors,50):.6f}")
    log.info(f"  Score range (negated): min={scores.min():.6f}  max={scores.max():.6f}")
    log.info("  Run evaluate.py to find the optimal threshold.")

    # Wrap with a placeholder threshold — evaluate.py will find the real one
    wrapper = AutoencoderWrapper(
        model=model.to("cpu"),   # always save on CPU for portability
        device="cpu",
        threshold=float(np.percentile(scores, 1)),  # p1 as starting point
    )
    return wrapper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-rf", action="store_true",
        help="Skip RF training and load existing classifier.pkl — only train autoencoder"
    )
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("CIC-IDS2018 — Model Training Pipeline")
    log.info("=" * 55)

    X_train, y_train, X_benign, feature_list, le = load_data()

    # Phase 5a — Random Forest
    clf_path = MODELS_DIR / "classifier.pkl"
    if args.skip_rf:
        if not clf_path.exists():
            raise FileNotFoundError(
                f"--skip-rf requested but {clf_path} not found. "
                "Run without --skip-rf first."
            )
        log.info("Skipping RF — loading existing classifier.pkl ...")
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        log.info(f"  Loaded classifier.pkl  (OOB={clf.oob_score_:.4f})")
    else:
        clf = train_classifier(X_train, y_train, le)
        save_model(clf, clf_path, "RandomForestClassifier")

    # Phase 5b — Autoencoder
    ae_wrapper = train_anomaly_detector(X_benign, clf, feature_list)
    save_model(ae_wrapper, MODELS_DIR / "anomaly.pkl", "AutoencoderWrapper")

    log.info("=" * 55)
    log.info("Training complete.")
    log.info(f"  Models saved → {MODELS_DIR}")
    log.info("  Next step: python src/evaluate.py")
    log.info("=" * 55)


if __name__ == "__main__":
    main()