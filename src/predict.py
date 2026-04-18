"""
predict.py — Runtime inference engine
======================================
Loads all artefacts once at startup, then exposes a single function:

    verdict = predict(flow_features: dict) -> Verdict

Called by:
    capture.py  — live nfstream flows
    api.py      — FastAPI endpoint
    tests/test_predict.py — smoke test
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
DEBUG = False
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# Configuration  (override via environment variables or .env)
# ---------------------------------------------------------------------------

# AE_THRESHOLD: reconstruction error above this → AE flags as anomalous.
# Derived from evaluate.py optimal F1predict() threshold sweep.
AE_THRESHOLD = float(os.getenv("AE_THRESHOLD", "0.003863"))

# RF_CONF_MIN: minimum RF confidence to treat RF label as reliable.
# Below this, RF label is still used but marked as low-confidence.
RF_CONF_MIN = float(os.getenv("RF_CONF_MIN", "0.50"))
RF_CONF_HIGH = float(os.getenv("RF_CONF_HIGH", "0.70"))  # NEW

# BENIGN_LABEL: must match LabelEncoder class name exactly.
BENIGN_LABEL = os.getenv("BENIGN_LABEL", "Benign")


# ---------------------------------------------------------------------------
# Verdict dataclass — what predict() returns
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    # RF result
    label: str         # e.g. "DDoS attacks-LOIC-HTTP", "Benign"
    confidence: float  # RF probability for predicted class [0, 1]
    is_attack: bool    # True if verdict is ATTACK or SUSPECT

    # AE result
    anomaly_score: float    # raw reconstruction error (higher = more suspicious)
    anomaly_flagged: bool   # True if reconstruction error > AE_THRESHOLD

    # Fusion decision
    verdict: str  # "ATTACK" | "SUSPECT" | "ANOMALY" | "BENIGN"
    source: str   # "RF+AE" | "RF" | "AE" | "none"

    # Per-class RF probabilities (for alert enrichment)
    class_probabilities: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label"              : self.label,
            "confidence"         : round(self.confidence, 4),
            "is_attack"          : self.is_attack,
            "anomaly_score"      : round(self.anomaly_score, 6),
            "anomaly_flagged"    : self.anomaly_flagged,
            "verdict"            : self.verdict,
            "source"             : self.source,
            "class_probabilities": {
                k: round(v, 4) for k, v in self.class_probabilities.items()
            },
        }


# ---------------------------------------------------------------------------
# MLEngine — loads artefacts once, runs inference on every flow
# ---------------------------------------------------------------------------
class MLEngine:
    """
    Singleton-style inference engine.
    Instantiate once at app startup, reuse for every flow.

    Usage:
        engine = MLEngine()
        verdict = engine.predict(flow_features)
    """

    def __init__(self) -> None:
        self._loaded       = False
        self.clf           = None   # RandomForestClassifier
        self.ae_wrapper    = None   # AutoencoderWrapper
        self.scaler        = None   # StandardScaler (RF)
        self.benign_scaler = None   # StandardScaler (AE, benign-only fitted)
        self.le            = None   # LabelEncoder
        self.feature_list: list[str] = []
        self.benign_class_id: int    = -1
    def load(self) -> None:
        """Load all artefacts from disk. Call once at startup."""
        if self._loaded:
            return

        log.info("MLEngine: loading artefacts ...")

        self.clf = self._load_pkl(MODELS_DIR / "classifier.pkl", "RandomForest")
        self.ae_wrapper = self._load_pkl(MODELS_DIR / "anomaly.pkl", "AutoencoderWrapper")
        self.scaler = self._load_pkl(MODELS_DIR / "scaler.pkl", "scaler")
        self.benign_scaler = self._load_pkl(MODELS_DIR / "benign_scaler.pkl", "benign_scaler")
        self.le = self._load_pkl(MODELS_DIR / "label_encoder.pkl", "LabelEncoder")

         
        self.scaler.set_output(transform="pandas")
        self.benign_scaler.set_output(transform="pandas")

        self.benign_class_id = list(self.le.classes_).index(BENIGN_LABEL)

        with open(DATA_DIR / "feature_list.json") as f:
            self.feature_list = json.load(f)

        log.info(
            f"MLEngine: ready — {len(self.feature_list)} features, "
            f"{len(self.le.classes_)} classes  |  "
            f"AE_THRESHOLD={AE_THRESHOLD}"
        )
        self._loaded = True
    # ------------------------------------------------------------------
    # Core inference — single flow
    # ------------------------------------------------------------------
    def predict(self, flow: dict) -> Verdict:
        """
        Run RF and AE simultaneously on a single flow, fuse with AND logic.

        Args:
            flow: dict mapping feature name → raw numeric value.
                  Missing keys are filled with 0. Extra keys are ignored.

        Returns:
            Verdict with full inference details.
        """
        if not self._loaded:
            self.load()

        # 1. Build raw feature vector once
        raw_vector = self._build_vector(flow)             # (1, 72)
        if DEBUG:
            print("[DEBUG] BEFORE RF scaler:", type(raw_vector))
        # 2. RF inference
        rf_array = self.scaler.transform(raw_vector)

        rf_vector = pd.DataFrame(
            rf_array,
            columns=self.feature_list
        )
        if DEBUG:
            print("[DEBUG] AFTER RF scaler:", type(rf_vector))
        rf_label, rf_confidence, rf_proba  = self._run_classifier(rf_vector)
        rf_is_attack                       = (rf_label != BENIGN_LABEL)

        # 3. AE inference
        if DEBUG:
            print("[DEBUG] BEFORE AE scaler:", type(raw_vector))
 
        ae_array = self.benign_scaler.transform(raw_vector)

        ae_vector = pd.DataFrame(
            ae_array,
            columns=self.feature_list
        )
        neg_score       = float(self.ae_wrapper.score_samples(ae_vector)[0])
        anomaly_score   = -neg_score                      # positive reconstruction error
        anomaly_flagged = anomaly_score > AE_THRESHOLD

        # 4. Fusion — AND logic
        verdict, source = self._fuse(rf_is_attack, rf_confidence, anomaly_flagged)
        is_attack       = verdict in ("ATTACK", "SUSPECT")

        return Verdict(
            label               = rf_label,
            confidence          = rf_confidence,
            is_attack           = is_attack,
            anomaly_score       = anomaly_score,
            anomaly_flagged     = anomaly_flagged,
            verdict             = verdict,
            source              = source,
            class_probabilities = rf_proba,
        )

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    def predict_batch(self, flows: list[dict]) -> list[Verdict]:
        """Run RF and AE on all flows simultaneously, fuse results."""
        if not self._loaded:
            self.load()

        if not flows:              #guard against empty input
            return []
        raw_matrix = pd.concat(
            [self._build_vector(f) for f in flows],
            ignore_index=True
        )

        if DEBUG:
            print("[DEBUG] BATCH raw_matrix type:", type(raw_matrix))
            print("[DEBUG] BATCH shape:", raw_matrix.shape) # (n, 72)

        # RF — full batch
        rf_array = self.scaler.transform(raw_matrix)
        rf_matrix = pd.DataFrame(rf_array, columns=self.feature_list)
        y_pred     = self.clf.predict(rf_matrix)
        y_proba    = self.clf.predict_proba(rf_matrix)                   # (n, n_classes)

        # AE — full batch
        ae_array = self.benign_scaler.transform(raw_matrix)
        ae_matrix = pd.DataFrame(ae_array, columns=self.feature_list)
        neg_scores   = self.ae_wrapper.score_samples(ae_matrix)          # (n,)
        ae_errors    = -neg_scores                                        # positive errors
        ae_flagged   = ae_errors > AE_THRESHOLD

        verdicts = []
        for i in range(len(flows)):
            cls_id      = int(y_pred[i])
            label       = self.le.classes_[cls_id]
            confidence  = float(y_proba[i][cls_id])
            proba_dict  = {self.le.classes_[j]: float(y_proba[i][j])
                           for j in range(len(self.le.classes_))}
            rf_is_attack = (label != BENIGN_LABEL)
            flagged      = bool(ae_flagged[i])

            verdict, source = self._fuse(rf_is_attack, confidence, flagged)
            verdicts.append(Verdict(
                label               = label,
                confidence          = confidence,
                is_attack           = verdict in ("ATTACK", "SUSPECT"),
                anomaly_score       = float(ae_errors[i]),
                anomaly_flagged     = flagged,
                verdict             = verdict,
                source              = source,
                class_probabilities = proba_dict,
            ))
        return verdicts

    # ------------------------------------------------------------------
    # Fusion logic
    # ------------------------------------------------------------------
    @staticmethod
    def _fuse(
        rf_is_attack: bool,
        rf_confidence: float,
        ae_flagged: bool,
    ) -> tuple[str, str]:
        """
        Confidence-aware fusion — returns (verdict, source).

     ┌──────────────┬──────────┬─────────────┬───────────┬──────────┐
     │ RF attack?   │ RF conf  │ AE flagged? │ Verdict   │ Source   │
     ├──────────────┼──────────┼─────────────┼───────────┼──────────┤
     │ Yes          │ ≥ 0.70   │ any         │ ATTACK    │ RF       │
     │ Yes          │ < 0.70   │ Yes         │ ATTACK    │ RF+AE    │
     │ Yes          │ < 0.70   │ No          │ SUSPECT   │ RF       │
     │ No           │ any      │ Yes         │ ANOMALY   │ AE       │
     │ No           │ any      │ No          │ BENIGN    │ RF+AE    │
     └──────────────┴──────────┴─────────────┴───────────┴──────────┘
      """
        if rf_is_attack:
            if rf_confidence >= RF_CONF_HIGH:
                return "ATTACK", "RF"
            elif ae_flagged:
                return "ATTACK", "RF+AE"
            else:
                return "SUSPECT", "RF"
        else:
            if ae_flagged:
                return "ANOMALY", "AE"
            else:
                return "BENIGN", "RF+AE"   
            
        

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_vector(self, flow: dict):
   

        row = {feat: float(flow.get(feat, 0.0)) for feat in self.feature_list}
        df = pd.DataFrame([row], columns=self.feature_list)

        if DEBUG:
            print("[DEBUG] _build_vector type:", type(df))
            print("[DEBUG] columns match:", list(df.columns) == self.feature_list)

        return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _run_classifier(self, rf_vector: np.ndarray) -> tuple[str, float, dict]:
        cls_id     = int(self.clf.predict(rf_vector)[0])
        label      = self.le.classes_[cls_id]
        proba      = self.clf.predict_proba(rf_vector)[0]
        confidence = float(proba[cls_id])
        proba_dict = {self.le.classes_[i]: float(proba[i])
                      for i in range(len(self.le.classes_))}
        return label, confidence, proba_dict

    @staticmethod


    def _load_pkl(path: Path, name: str):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}. Run train.py first.")

        obj = joblib.load(path)  # handles compressed sklearn models
        size_mb = path.stat().st_size / 1e6
        log.info(f"  Loaded {name:20s} ← {path.name}  ({size_mb:.1f} MB)")
        return obj
# ---------------------------------------------------------------------------
# Module-level singleton — used by capture.py and api.py
# ---------------------------------------------------------------------------
_engine: Optional[MLEngine] = None

def get_engine() -> MLEngine:
    """Return the shared MLEngine instance, loading on first call."""
    global _engine
    if _engine is None:
        _engine = MLEngine()
        _engine.load()
    return _engine


# ---------------------------------------------------------------------------
# Smoke test  (python src/predict.py)
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("=" * 55)
    log.info("predict.py — Smoke Test")
    log.info("=" * 55)

    engine = get_engine()

    # Test 1 — zero flow
    zero_flow = {f: 0.0 for f in engine.feature_list}
    v = engine.predict(zero_flow)
    log.info(f"\nTest 1 — Zero flow:")
    log.info(f"  verdict={v.verdict}  label={v.label}  "
             f"conf={v.confidence:.4f}  ae_score={v.anomaly_score:.6f}  "
             f"ae_flagged={v.anomaly_flagged}  source={v.source}")

    # Test 2 — real flows from test_raw.pkl
    import pandas as pd
    test_path = DATA_DIR / "processed" / "test_raw.pkl"
    if test_path.exists():
        test_df = pd.read_pickle(test_path)
        sample  = test_df.sample(n=min(10, len(test_df)), random_state=42)
        log.info(f"\nTest 2 — 10 real flows from test_raw.pkl:")
        log.info(f"  {'true label':35s}  {'pred label':35s}  "
                 f"{'conf':5s}  {'ae_flag':7s}  {'verdict':7s}  source")
        log.info("  " + "-" * 105)
        for _, row in sample.iterrows():
            true_label = row.get("Label", "unknown")
            flow_dict  = {f: row[f] for f in engine.feature_list if f in row}
            v = engine.predict(flow_dict)
            log.info(
                f"  {true_label:35s}  {v.label:35s}  "
                f"{v.confidence:.3f}  "
                f"{'YES' if v.anomaly_flagged else 'no':7s}  "
                f"{v.verdict:7s}  {v.source}"
            )
    else:
        log.warning("  test_raw.pkl not found — skipping real-flow test")

    # Test 3 — batch predict
    batch  = [{f: 0.0 for f in engine.feature_list} for _ in range(3)]
    result = engine.predict_batch(batch)
    log.info(f"\nTest 3 — Batch predict (3 zero flows):")
    for i, v in enumerate(result):
        log.info(f"  [{i}] verdict={v.verdict}  source={v.source}  "
                 f"ae_score={v.anomaly_score:.6f}")

    # Test 4 — fusion logic explanation
    log.info(f"\nTest 4 — Fusion logic (simultaneous AND):")
    log.info(f"  RF attack  + AE flags   → ATTACK  (block)")
    log.info(f"  RF attack  + AE passes  → SUSPECT (alert only)")
    log.info(f"  RF benign  + AE flags   → ANOMALY (alert + rate limit)")
    log.info(f"  RF benign  + AE passes  → BENIGN  (pass through)")
    log.info(f"  AE_THRESHOLD = {AE_THRESHOLD}")

    log.info("\n" + "=" * 55)
    log.info("Smoke test PASSED ✓")
    log.info("=" * 55)
    log.info("Next step: python src/api.py")


if __name__ == "__main__":
    _smoke_test()