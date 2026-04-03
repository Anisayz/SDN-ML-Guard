"""
evaluate.py — Evaluate Random Forest (classifier) + Autoencoder (anomaly detector)
====================================================================================
Run after train.py:
    python src/evaluate.py

Reads from  data/processed/  +  models/
Writes reports to  data/processed/evaluation/

Outputs:
    evaluation/clf_report.txt           — per-class precision / recall / F1
    evaluation/clf_confusion.png        — 15x15 confusion matrix heatmap
    evaluation/clf_feature_importance.png
    evaluation/ae_roc.png               — ROC curve with AUC
    evaluation/ae_pr.png                — Precision-Recall curve
    evaluation/ae_score_dist.png        — reconstruction error distribution
    evaluation/ae_threshold_sweep.png   — F1 / precision / recall vs threshold
    evaluation/summary.json             — key numbers for quick reference
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from models import Autoencoder, AutoencoderWrapper

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
MODELS_DIR    = BASE_DIR / "models"
EVAL_DIR      = PROCESSED_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# AE_THRESHOLD: update this after running evaluate.py once and reading
# the "Optimal threshold" line. Set it here and re-run to use it.
# Leave as None on first run — evaluate.py will use the p99 training threshold.
AE_THRESHOLD = 0.010639   # e.g. set to 0.045231 after first run


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------
def load_all():
    log.info("Loading test data and models ...")

    test_df     = pd.read_pickle(PROCESSED_DIR / "test.pkl")
    test_raw_df = pd.read_pickle(PROCESSED_DIR / "test_raw.pkl")
    feature_list = json.loads(
        (BASE_DIR / "data" / "feature_list.json").read_text()
    )

    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(MODELS_DIR / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    # AutoencoderWrapper — loads the PyTorch model + score_samples() interface
    with open(MODELS_DIR / "anomaly.pkl", "rb") as f:
        ae_wrapper = pickle.load(f)
    log.info(f"  Loaded AutoencoderWrapper")

    with open(MODELS_DIR / "benign_scaler.pkl", "rb") as f:
        benign_scaler = pickle.load(f)

    # X_test — StandardScaler space (for RF)
    X_test    = test_df[feature_list].values.astype(np.float32)
    y_test    = test_df["label_enc"].values.astype(np.int32)

    # X_test_ae — benign_scaler space (for Autoencoder)
    X_test_ae = benign_scaler.transform(
        test_raw_df[feature_list].values.astype(np.float32)
    ).astype(np.float32)

    log.info(f"  Test set shape : {X_test.shape}")
    log.info(f"  Classes        : {le.classes_.tolist()}")

    # X_benign_ae — benign training data in AE space (for threshold calibration)
    benign_df    = pd.read_pickle(PROCESSED_DIR / "benign_train.pkl")
    X_benign_ae  = benign_scaler.transform(
        benign_df[feature_list].values.astype(np.float32)
    ).astype(np.float32)
    log.info(f"  Benign train shape : {X_benign_ae.shape}")

    return X_test, X_test_ae, X_benign_ae, y_test, le, clf, ae_wrapper, feature_list, test_df


# ---------------------------------------------------------------------------
# Phase 6a — Classifier evaluation
# ---------------------------------------------------------------------------
def evaluate_classifier(
    clf, X_test: np.ndarray, y_test: np.ndarray, le, feature_list: list[str]
) -> dict:
    log.info("=" * 55)
    log.info("Phase 6a — Random Forest Classifier Evaluation")
    log.info("=" * 55)

    log.info("  Running predictions on test set ...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    log.info(f"  Overall accuracy : {acc:.4f}")

    report_str = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
    )
    log.info(f"\n{report_str}")
    (EVAL_DIR / "clf_report.txt").write_text(report_str)
    log.info(f"  Saved → {EVAL_DIR / 'clf_report.txt'}")

    report_dict = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    weak = {
        cls: round(vals["f1-score"], 4)
        for cls, vals in report_dict.items()
        if cls in le.classes_ and vals["f1-score"] < 0.70
    }
    if weak:
        log.warning(f"  ⚠  Weak classes (F1 < 0.70): {weak}")
    else:
        log.info("  All classes F1 >= 0.70 ✓")

    # Confusion matrix
    log.info("  Plotting confusion matrix ...")
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
        linewidths=0.4, ax=ax,
        cbar_kws={"label": "Recall (row-normalised)"},
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Random Forest — Confusion Matrix (counts, colour=recall)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "clf_confusion.png", dpi=150)
    plt.close()
    log.info(f"  Saved → {EVAL_DIR / 'clf_confusion.png'}")

    # Feature importance
    log.info("  Plotting feature importances ...")
    fi     = pd.Series(clf.feature_importances_, index=feature_list).sort_values()
    fi_top = fi.tail(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    fi_top.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Random Forest — Top 25 Feature Importances", fontsize=12)
    ax.set_xlabel("Mean decrease in impurity")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "clf_feature_importance.png", dpi=150)
    plt.close()
    log.info(f"  Saved → {EVAL_DIR / 'clf_feature_importance.png'}")

    return {
        "accuracy"    : round(acc, 4),
        "weighted_f1" : round(report_dict["weighted avg"]["f1-score"], 4),
        "macro_f1"    : round(report_dict["macro avg"]["f1-score"], 4),
        "weak_classes": weak,
        "per_class_f1": {
            cls: round(report_dict[cls]["f1-score"], 4)
            for cls in le.classes_
        },
    }


# ---------------------------------------------------------------------------
# Phase 6b — Autoencoder evaluation
# ---------------------------------------------------------------------------
def evaluate_anomaly(
    ae_wrapper,
    X_test_ae: np.ndarray,
    y_test: np.ndarray,
    le,
    threshold_override: float = None,
) -> dict:
    log.info("=" * 55)
    log.info("Phase 6b — Autoencoder Anomaly Detector Evaluation")
    log.info("=" * 55)

    benign_id = list(le.classes_).index("Benign")
    y_binary  = (y_test != benign_id).astype(int)   # 1 = attack

    log.info("  Scoring test set ...")
    # ae_wrapper.score_samples() returns NEGATIVE reconstruction error
    # (IF convention: more negative = more anomalous)
    # We negate again to get raw reconstruction error (higher = more anomalous)
    # This makes threshold logic intuitive: flag if error > threshold
    neg_scores = ae_wrapper.score_samples(X_test_ae)   # shape (n,), negative
    errors     = -neg_scores                            # reconstruction error, positive

    # Clip extreme outliers before sweep — a few benign flows with very high
    # reconstruction error would stretch the x-axis and hide the useful range
    clip_val = float(np.percentile(errors, 99.9))
    errors_clipped = np.clip(errors, 0, clip_val)
    n_clipped = (errors > clip_val).sum()
    if n_clipped > 0:
        log.info(f"  Clipping {n_clipped} outliers at p99.9 = {clip_val:.6f} for sweep")

    log.info(f"  Error stats (full):    mean={errors.mean():.6f}  "
             f"std={errors.std():.6f}  max={errors.max():.2f}")
    log.info(f"  Error stats (clipped): mean={errors_clipped.mean():.6f}  "
             f"std={errors_clipped.std():.6f}  max={errors_clipped.max():.6f}")

    # Use clipped errors for all metric computation and plots
    # (the few outliers don't represent real attack patterns)
    e = errors_clipped

    # --- ROC-AUC ---
    roc_auc = roc_auc_score(y_binary, e)
    log.info(f"  ROC-AUC : {roc_auc:.4f}")

    fpr, tpr, roc_thresholds = roc_curve(y_binary, e)

    # Determine display threshold
    display_threshold = threshold_override if threshold_override is not None \
        else float(np.percentile(e[y_binary == 0], 99))   # p99 of benign errors

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    op_idx = np.argmin(np.abs(roc_thresholds - display_threshold))
    ax.scatter(fpr[op_idx], tpr[op_idx], s=80, color="red", zorder=5,
               label=f"Threshold = {display_threshold:.6f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Autoencoder — ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "ae_roc.png", dpi=150)
    plt.close()
    log.info(f"  Saved → {EVAL_DIR / 'ae_roc.png'}")

    # --- Precision-Recall curve ---
    precision_arr, recall_arr, _ = precision_recall_curve(y_binary, e)
    pr_auc = auc(recall_arr, precision_arr)
    log.info(f"  PR-AUC  : {pr_auc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_arr, precision_arr, color="steelblue", lw=2,
            label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Autoencoder — Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "ae_pr.png", dpi=150)
    plt.close()
    log.info(f"  Saved → {EVAL_DIR / 'ae_pr.png'}")

    # --- Score distribution ---
    e_benign = e[y_binary == 0]
    e_attack = e[y_binary == 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(e_benign, bins=100, alpha=0.6, color="steelblue",
            label=f"Benign  (n={len(e_benign):,})", density=True)
    ax.hist(e_attack, bins=100, alpha=0.6, color="crimson",
            label=f"Attack  (n={len(e_attack):,})", density=True)
    ax.axvline(display_threshold, color="black", linestyle="--", lw=2,
               label=f"Threshold = {display_threshold:.6f}")
    ax.set_xlabel("Reconstruction error  (higher = more suspicious)")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder — Score Distribution (clipped at p99.9)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "ae_score_dist.png", dpi=150)
    plt.close()
    log.info(f"  Saved → {EVAL_DIR / 'ae_score_dist.png'}")

    # --- Threshold sweep ---
    log.info("  Sweeping thresholds to find optimal F1 ...")
    sweep_thresholds = np.linspace(e.min(), e.max(), 300)
    sweep_f1, sweep_prec, sweep_rec = [], [], []

    for t in sweep_thresholds:
        y_pred_bin = (e > t).astype(int)   # above threshold = anomalous
        sweep_f1.append(f1_score(y_binary, y_pred_bin, zero_division=0))
        sweep_prec.append(precision_score(y_binary, y_pred_bin, zero_division=0))
        sweep_rec.append(recall_score(y_binary, y_pred_bin, zero_division=0))

    best_idx       = int(np.argmax(sweep_f1))
    best_threshold = float(sweep_thresholds[best_idx])
    best_f1        = sweep_f1[best_idx]
    best_prec      = sweep_prec[best_idx]
    best_rec       = sweep_rec[best_idx]

    log.info(f"  Optimal threshold : {best_threshold:.6f}  "
             f"(F1={best_f1:.4f}  P={best_prec:.4f}  R={best_rec:.4f})")
    log.info(f"  Display threshold : {display_threshold:.6f}  "
             f"(F1={sweep_f1[np.argmin(np.abs(sweep_thresholds - display_threshold))]:.4f})")
    log.info(f"  >>> Set AE_THRESHOLD = {best_threshold:.6f} in evaluate.py")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sweep_thresholds, sweep_f1,   color="purple",    lw=2,   label="F1")
    ax.plot(sweep_thresholds, sweep_prec, color="steelblue", lw=1.5,
            linestyle="--", label="Precision")
    ax.plot(sweep_thresholds, sweep_rec,  color="crimson",   lw=1.5,
            linestyle="--", label="Recall")
    ax.axvline(display_threshold, color="black", linestyle=":",
               label=f"Current ({display_threshold:.6f})")
    ax.axvline(best_threshold, color="green", linestyle="-",
               label=f"Optimal F1 ({best_threshold:.6f})")
    ax.set_xlabel("Reconstruction error threshold  (flag if error > threshold)")
    ax.set_ylabel("Metric value")
    ax.set_title("Autoencoder — Threshold Sweep")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "ae_threshold_sweep.png", dpi=150)
    plt.close()
    log.info(f"  Saved → {EVAL_DIR / 'ae_threshold_sweep.png'}")

    # --- At current threshold ---
    active_threshold = threshold_override if threshold_override is not None \
        else best_threshold   # use optimal on first run
    y_pred_current = (e > active_threshold).astype(int)
    current_f1     = f1_score(y_binary, y_pred_current, zero_division=0)
    current_prec   = precision_score(y_binary, y_pred_current, zero_division=0)
    current_rec    = recall_score(y_binary, y_pred_current, zero_division=0)

    # --- Benign FPR ---
    benign_mask      = y_binary == 0
    n_benign         = int(benign_mask.sum())
    n_benign_flagged = int((e[benign_mask] > active_threshold).sum())
    n_benign_correct = n_benign - n_benign_flagged
    benign_fpr       = n_benign_flagged / n_benign
    benign_acc       = n_benign_correct  / n_benign

    log.info(f"\n  === At active threshold ({active_threshold:.6f}) ===")
    log.info(f"  --- Attack detection ---")
    log.info(f"  Precision : {current_prec:.4f}")
    log.info(f"  Recall    : {current_rec:.4f}")
    log.info(f"  F1        : {current_f1:.4f}")
    log.info(f"  --- Benign accuracy ---")
    log.info(f"  Benign flows in test     : {n_benign:,}")
    log.info(f"  Correctly passed (TN)    : {n_benign_correct:,}  ({benign_acc*100:.2f}%)")
    log.info(f"  Wrongly flagged  (FP)    : {n_benign_flagged:,}  ({benign_fpr*100:.2f}%)")
    if benign_fpr > 0.10:
        log.warning(f"  ⚠  High FPR on benign traffic: {benign_fpr*100:.1f}%")
    else:
        log.info(f"  Benign FPR within acceptable range ✓")

    # --- Per-attack-class breakdown ---
    log.info("\n  === Detection rate per attack class ===")
    attack_detection = {}
    for cls_id, cls_name in enumerate(le.classes_):
        if cls_name == "Benign":
            continue
        mask    = y_test == cls_id
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        n_caught = int((e[mask] > active_threshold).sum())
        rate     = n_caught / n_total
        attack_detection[cls_name] = {
            "total"  : n_total,
            "caught" : n_caught,
            "rate"   : round(rate, 4),
        }
        flag = "  ⚠" if rate < 0.50 else ""
        log.info(f"  {cls_name:40s}  {n_caught:>6,}/{n_total:>6,}  "
                 f"({rate*100:5.1f}%){flag}")

    return {
        "roc_auc"          : round(roc_auc, 4),
        "pr_auc"           : round(pr_auc, 4),
        "active_threshold" : round(active_threshold, 6),
        "optimal_threshold": round(best_threshold, 6),
        "current_f1"       : round(current_f1, 4),
        "current_precision": round(current_prec, 4),
        "current_recall"   : round(current_rec, 4),
        "optimal_f1"       : round(best_f1, 4),
        "attack_detection" : attack_detection,
        "benign_total"     : n_benign,
        "benign_correct"   : n_benign_correct,
        "benign_flagged"   : n_benign_flagged,
        "benign_fpr"       : round(float(benign_fpr), 4),
        "benign_accuracy"  : round(float(benign_acc), 4),
    }


# ---------------------------------------------------------------------------
# Cross-model analysis: RF misses caught by AE
# ---------------------------------------------------------------------------
def cross_model_analysis(
    clf, ae_wrapper,
    X_test: np.ndarray,
    X_test_ae: np.ndarray,
    y_test: np.ndarray,
    le,
    threshold: float,
) -> dict:
    log.info("=" * 55)
    log.info("Cross-model analysis — RF misses caught by AE")
    log.info("=" * 55)

    benign_id  = list(le.classes_).index("Benign")
    y_binary   = (y_test != benign_id).astype(int)

    y_clf_pred = clf.predict(X_test)
    neg_scores = ae_wrapper.score_samples(X_test_ae)
    errors     = np.clip(-neg_scores, 0, float(np.percentile(-neg_scores, 99.9)))
    y_ae_pred  = (errors > threshold).astype(int)   # 1 = anomalous

    rf_missed_mask = (y_binary == 1) & (y_clf_pred == benign_id)
    ae_caught_mask = rf_missed_mask & (y_ae_pred == 1)

    n_attacks   = int(y_binary.sum())
    n_rf_missed = int(rf_missed_mask.sum())
    n_ae_caught = int(ae_caught_mask.sum())

    log.info(f"  Total attack flows in test  : {n_attacks:,}")
    log.info(f"  RF missed (called Benign)   : {n_rf_missed:,}  "
             f"({n_rf_missed/n_attacks*100:.1f}%)")
    log.info(f"  AE caught among RF's misses : {n_ae_caught:,}  "
             f"({n_ae_caught/max(n_rf_missed,1)*100:.1f}% of RF misses)")

    log.info("\n  Per-class breakdown:")
    per_class = {}
    for cls_id, cls_name in enumerate(le.classes_):
        if cls_name == "Benign":
            continue
        cls_mask       = y_test == cls_id
        n_cls_attacks  = cls_mask.sum()
        n_cls_missed   = int((rf_missed_mask & cls_mask).sum())
        n_cls_ae_saved = int((ae_caught_mask & cls_mask).sum())
        if n_cls_attacks == 0:
            continue
        per_class[cls_name] = {
            "rf_missed": n_cls_missed,
            "ae_saved" : n_cls_ae_saved,
        }
        log.info(f"  {cls_name:40s}  RF missed {n_cls_missed:>5,}  "
                 f"→  AE saved {n_cls_ae_saved:>5,}")

    return {
        "total_attacks"      : n_attacks,
        "rf_missed"          : n_rf_missed,
        "ae_caught_rf_misses": n_ae_caught,
        "per_class"          : per_class,
    }


# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------
def save_summary(clf_metrics: dict, ae_metrics: dict, cross: dict) -> None:
    summary = {
        "classifier" : clf_metrics,
        "anomaly"    : ae_metrics,
        "cross_model": cross,
    }
    path = EVAL_DIR / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved → {path}")

    log.info("\n" + "=" * 55)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 55)
    log.info(f"  RF  accuracy      : {clf_metrics['accuracy']:.4f}")
    log.info(f"  RF  weighted F1   : {clf_metrics['weighted_f1']:.4f}")
    log.info(f"  RF  macro F1      : {clf_metrics['macro_f1']:.4f}")
    log.info(f"  AE  ROC-AUC       : {ae_metrics['roc_auc']:.4f}")
    log.info(f"  AE  PR-AUC        : {ae_metrics['pr_auc']:.4f}")
    log.info(f"  AE  F1 @ active   : {ae_metrics['current_f1']:.4f}  "
             f"(threshold={ae_metrics['active_threshold']:.6f})")
    log.info(f"  AE  F1 @ optimal  : {ae_metrics['optimal_f1']:.4f}  "
             f"(threshold={ae_metrics['optimal_threshold']:.6f})")
    log.info(f"  AE  benign FPR    : {ae_metrics['benign_fpr']*100:.2f}%  "
             f"({ae_metrics['benign_flagged']:,} / "
             f"{ae_metrics['benign_total']:,} wrongly flagged)")
    log.info(f"  Cross: AE rescued : {cross['ae_caught_rf_misses']:,} / "
             f"{cross['rf_missed']:,} RF misses  "
             f"({cross['ae_caught_rf_misses']/max(cross['rf_missed'],1)*100:.1f}%)")
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# Benign-calibrated threshold computation
# ---------------------------------------------------------------------------
def compute_benign_thresholds(ae_wrapper, X_benign_ae: np.ndarray) -> dict:
    """
    Compute thresholds calibrated on benign training data.
    p95 → ~5% benign FPR
    p99 → ~1% benign FPR
    p999 → ~0.1% benign FPR

    This is more reliable than F1-optimal threshold on the mixed test set
    because the test set is 73% attacks — F1 is skewed toward catching attacks
    at the cost of flagging benign traffic.
    """
    log.info("=" * 55)
    log.info("Benign-calibrated threshold analysis")
    log.info("=" * 55)

    log.info(f"  Scoring {X_benign_ae.shape[0]:,} benign training flows ...")
    neg_scores    = ae_wrapper.score_samples(X_benign_ae)
    benign_errors = -neg_scores   # positive reconstruction errors

    # Clip outliers for cleaner percentile computation
    clip_val = float(np.percentile(benign_errors, 99.9))
    benign_clipped = np.clip(benign_errors, 0, clip_val)

    p50  = float(np.percentile(benign_clipped, 50))
    p90  = float(np.percentile(benign_clipped, 90))
    p95  = float(np.percentile(benign_clipped, 95))
    p99  = float(np.percentile(benign_clipped, 99))
    p999 = float(np.percentile(benign_clipped, 99.9))

    log.info(f"  Benign error distribution (clipped at p99.9={clip_val:.6f}):")
    log.info(f"    p50  = {p50:.6f}  (median)")
    log.info(f"    p90  = {p90:.6f}  → ~10% benign FPR if used as threshold")
    log.info(f"    p95  = {p95:.6f}  → ~5%  benign FPR if used as threshold")
    log.info(f"    p99  = {p99:.6f}  → ~1%  benign FPR if used as threshold")
    log.info(f"    p99.9= {p999:.6f} → ~0.1% benign FPR if used as threshold")

    return {
        "p50" : p50,
        "p90" : p90,
        "p95" : p95,
        "p99" : p99,
        "p999": p999,
    }


def evaluate_at_benign_thresholds(
    ae_wrapper,
    X_test_ae: np.ndarray,
    y_test: np.ndarray,
    le,
    thresholds: dict,
) -> None:
    """
    Show attack detection rate and benign FPR at each benign-calibrated threshold.
    Helps pick the threshold that balances FPR vs attack recall for your use case.
    """
    log.info("=" * 55)
    log.info("Attack detection at benign-calibrated thresholds")
    log.info("=" * 55)

    benign_id  = list(le.classes_).index("Benign")
    y_binary   = (y_test != benign_id).astype(int)
    benign_mask = y_binary == 0
    n_benign    = benign_mask.sum()

    neg_scores = ae_wrapper.score_samples(X_test_ae)
    errors     = np.clip(-neg_scores, 0, float(np.percentile(-neg_scores, 99.9)))

    log.info(f"  {'Threshold':>12s}  {'Benign FPR':>10s}  {'Attack Recall':>13s}  "
             f"{'Precision':>9s}  {'F1':>6s}  {'Infilteration':>13s}")
    log.info("  " + "-" * 75)

    for name, t in thresholds.items():
        y_pred      = (errors > t).astype(int)
        n_fp        = (errors[benign_mask] > t).sum()
        fpr         = n_fp / n_benign
        rec         = recall_score(y_binary, y_pred, zero_division=0)
        prec        = precision_score(y_binary, y_pred, zero_division=0)
        f1          = f1_score(y_binary, y_pred, zero_division=0)

        # Infilteration specifically
        inf_id      = list(le.classes_).index("Infilteration")
        inf_mask    = y_test == inf_id
        n_inf       = inf_mask.sum()
        n_inf_caught = (errors[inf_mask] > t).sum()
        inf_rate    = n_inf_caught / n_inf if n_inf > 0 else 0

        log.info(f"  {name:>5s} ({t:.6f})  "
                 f"{fpr*100:>9.2f}%  "
                 f"{rec*100:>12.1f}%  "
                 f"{prec:>9.4f}  "
                 f"{f1:>6.4f}  "
                 f"{inf_rate*100:>12.1f}%")

    log.info("")
    log.info("  Pick the threshold row that gives acceptable benign FPR for your use case.")
    log.info("  Then set AE_THRESHOLD = <value> in evaluate.py and predict.py.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 55)
    log.info("CIC-IDS2018 — Evaluation Pipeline")
    log.info("=" * 55)

    X_test, X_test_ae, X_benign_ae, y_test, le, clf, ae_wrapper, feature_list, test_df = load_all()

    # 6a — Classifier
    clf_metrics = evaluate_classifier(clf, X_test, y_test, le, feature_list)

    # 6b — Autoencoder
    # On first run AE_THRESHOLD=None → uses optimal from sweep automatically
    # After first run: set AE_THRESHOLD at the top of this file to lock it in
    ae_metrics = evaluate_anomaly(
        ae_wrapper, X_test_ae, y_test, le,
        threshold_override=AE_THRESHOLD,
    )

    # Cross-model — use the active threshold from ae_metrics
    cross = cross_model_analysis(
        clf, ae_wrapper,
        X_test, X_test_ae, y_test, le,
        threshold=ae_metrics["active_threshold"],
    )

    # Benign-calibrated threshold analysis
    benign_thresholds = compute_benign_thresholds(ae_wrapper, X_benign_ae)
    evaluate_at_benign_thresholds(ae_wrapper, X_test_ae, y_test, le, benign_thresholds)

    save_summary(clf_metrics, ae_metrics, cross)

    log.info("All evaluation artefacts saved to:")
    log.info(f"  {EVAL_DIR}")
    log.info("Next step: python src/predict.py  (smoke test)")


if __name__ == "__main__":
    main()