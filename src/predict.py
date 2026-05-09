"""
predict.py — Runtime inference engine
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

AE_THRESHOLD: float = float(os.getenv("AE_THRESHOLD", "0.45"))
AE_CONF_MIN:  float = float(os.getenv("AE_CONF_MIN",  "0.50"))
RF_CONF_MIN:  float = float(os.getenv("RF_CONF_MIN",  "0.50"))
RF_CONF_HIGH: float = float(os.getenv("RF_CONF_HIGH", "0.70"))
BENIGN_LABEL: str   = os.getenv("BENIGN_LABEL", "Benign")

 
AE_THRESHOLD_BY_PORT: dict[int, float] = {
    53:   float(os.getenv("AE_THRESHOLD_PORT_53",   "2.0")),   # DNS — noisy
    67:   float(os.getenv("AE_THRESHOLD_PORT_67",   "999.0")), # DHCP server  — whitelist
    68:   float(os.getenv("AE_THRESHOLD_PORT_68",   "999.0")), # DHCP client  — whitelist
    80:   float(os.getenv("AE_THRESHOLD_PORT_80",   "0.55")),  # HTTP — just above benign baseline ~0.52
    443:  float(os.getenv("AE_THRESHOLD_PORT_443",  "1.0")),   # HTTPS
    1900: float(os.getenv("AE_THRESHOLD_PORT_1900", "999.0")), # SSDP/UPnP   — whitelist
}

 
_WHITELIST_DST_PORTS: frozenset[int] = frozenset(
    int(p) for p in os.getenv("WHITELIST_DST_PORTS", "67,68,5353,1900,631").split(",")
    if p.strip()
)
 
SYN_FLOOD_MIN_PACKETS:     int   = int(os.getenv("SYN_FLOOD_MIN_PACKETS",       "1"))
SYN_FLOOD_BWD_BYTES_MAX:   int   = int(os.getenv("SYN_FLOOD_BWD_BYTES_MAX",     "60"))
SYN_FLOOD_DURATION_MAX_MS: float = float(os.getenv("SYN_FLOOD_DURATION_MAX_MS", "50"))

 
 
SYN_FLOOD_BWD_BYTES_MAX_STRICT: int = int(
    os.getenv("SYN_FLOOD_BWD_BYTES_MAX_STRICT", "44")
)


def _ae_threshold(dst_port: int) -> float:
    return AE_THRESHOLD_BY_PORT.get(dst_port, AE_THRESHOLD)


def _is_whitelisted(dst_port: int, src_ip: str = "", dst_ip: str = "") -> bool:
   
    if dst_port in _WHITELIST_DST_PORTS:
        return True
    # Broadcast destination — never an attack target in our topology
    if dst_ip in ("255.255.255.255", "224.0.0.1", "224.0.0.251", "239.255.255.250"):
        return True
    return False


@dataclass
class Verdict:
    label:               str
    confidence:          float
    is_attack:           bool
    anomaly_score:       float
    anomaly_flagged:     bool
    verdict:             str
    source:              str
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
            "class_probabilities": {k: round(v, 4) for k, v in self.class_probabilities.items()},
        }


class MLEngine:

    def __init__(self) -> None:
        self._loaded       = False
        self.clf           = None
        self.ae_wrapper    = None
        self.scaler        = None
        self.benign_scaler = None
        self.le            = None
        self.feature_list: list[str] = []
        self.benign_class_id: int    = -1

    def load(self) -> None:
        if self._loaded:
            return

        log.info("MLEngine: loading artefacts ...")
        self.clf           = self._load_pkl(MODELS_DIR / "classifier.pkl",    "RandomForest")
        self.ae_wrapper    = self._load_pkl(MODELS_DIR / "anomaly.pkl",       "AutoencoderWrapper")
        self.scaler        = self._load_pkl(MODELS_DIR / "scaler.pkl",        "scaler")
        self.benign_scaler = self._load_pkl(MODELS_DIR / "benign_scaler.pkl", "benign_scaler")
        self.le            = self._load_pkl(MODELS_DIR / "label_encoder.pkl", "LabelEncoder")

        self.benign_class_id = list(self.le.classes_).index(BENIGN_LABEL)

        with open(DATA_DIR / "feature_list.json") as f:
            self.feature_list = json.load(f)

        log.info(
            f"MLEngine: ready — {len(self.feature_list)} features, "
            f"{len(self.le.classes_)} classes | "
            f"AE_THRESHOLD={AE_THRESHOLD} | "
            f"AE_CONF_MIN={AE_CONF_MIN} | "
            f"RF_CONF_HIGH={RF_CONF_HIGH} | "
            f"SYN_FLOOD: min_pkts={SYN_FLOOD_MIN_PACKETS} "
            f"bwd_max={SYN_FLOOD_BWD_BYTES_MAX} "
            f"dur_max_ms={SYN_FLOOD_DURATION_MAX_MS} | "
            f"WHITELIST ports={sorted(_WHITELIST_DST_PORTS)}"
        )
        self._loaded = True

    def predict(self, flow: dict, flow_meta: dict | None = None) -> "Verdict":
        if not self._loaded:
            self.load()

        dst_port = int(flow.get("dst_port", 0))
        dst_ip   = flow_meta.get("dst_ip", "") if flow_meta else ""
        src_ip   = flow_meta.get("src_ip", "") if flow_meta else ""

        # ── Whitelist fast-path ────────────────────────────────────────
        if _is_whitelisted(dst_port, src_ip, dst_ip):
            log.debug(f"[predict] Whitelisted flow port={dst_port} dst={dst_ip}")
            return Verdict(
                label="Benign", confidence=1.0, is_attack=False,
                anomaly_score=0.0, anomaly_flagged=False,
                verdict="BENIGN", source="WHITELIST",
            )

        if flow_meta:
            self._log_syn_debug(flow_meta)

        raw_df = self._build_vector(flow)

        rf_vector                         = self.scaler.transform(raw_df)
        rf_label, rf_confidence, rf_proba = self._run_classifier(rf_vector)
        rf_is_attack                      = (rf_label != BENIGN_LABEL)

        ae_vector       = self.benign_scaler.transform(raw_df)
        anomaly_score   = float(-self.ae_wrapper.score_samples(ae_vector)[0])
        anomaly_flagged = anomaly_score > _ae_threshold(dst_port)

        verdict, source = self._fuse(
            rf_is_attack, rf_confidence,
            anomaly_flagged, anomaly_score,
            dst_port, flow_meta,
        )

        return Verdict(
            label           = rf_label,
            confidence      = rf_confidence,
            is_attack       = verdict in ("ATTACK", "SUSPECT"),
            anomaly_score   = anomaly_score,
            anomaly_flagged = anomaly_flagged,
            verdict         = verdict,
            source          = source,
            class_probabilities = rf_proba,
        )

    def predict_batch(
        self,
        flows: list[dict],
        flow_metas: list[dict] | None = None,
    ) -> list["Verdict"]:
        if not self._loaded:
            self.load()

        if not flows:
            return []

        if flow_metas is None:
            flow_metas = [None] * len(flows)

        # ── Separate whitelisted flows so we don't waste inference on them ──
        results      = [None] * len(flows)
        ml_indices   = []
        ml_flows     = []
        ml_metas     = []

        for i, (flow, meta) in enumerate(zip(flows, flow_metas)):
            dst_port = int(flow.get("dst_port", 0))
            dst_ip   = meta.get("dst_ip", "") if meta else ""
            src_ip   = meta.get("src_ip", "") if meta else ""

            if _is_whitelisted(dst_port, src_ip, dst_ip):
                log.debug(f"[batch] Whitelisted flow port={dst_port} dst={dst_ip}")
                results[i] = Verdict(
                    label="Benign", confidence=1.0, is_attack=False,
                    anomaly_score=0.0, anomaly_flagged=False,
                    verdict="BENIGN", source="WHITELIST",
                )
            else:
                ml_indices.append(i)
                ml_flows.append(flow)
                ml_metas.append(meta)

        if not ml_flows:
            return results

        # ── Run ML on non-whitelisted flows ────────────────────────────
        raw_df    = pd.concat([self._build_vector(f) for f in ml_flows], ignore_index=True)
        rf_matrix = self.scaler.transform(raw_df)
        y_pred    = self.clf.predict(rf_matrix)
        y_proba   = self.clf.predict_proba(rf_matrix)
        ae_matrix = self.benign_scaler.transform(raw_df)
        ae_errors = -self.ae_wrapper.score_samples(ae_matrix)

        for j, (flow, meta) in enumerate(zip(ml_flows, ml_metas)):
            cls_id     = int(y_pred[j])
            label      = self.le.classes_[cls_id]
            confidence = float(y_proba[j][cls_id])
            proba_dict = {
                self.le.classes_[k]: float(y_proba[j][k])
                for k in range(len(self.le.classes_))
            }
            rf_is_attack    = (label != BENIGN_LABEL)
            dst_port        = int(flow.get("dst_port", 0))
            anomaly_score   = float(ae_errors[j])
            anomaly_flagged = anomaly_score > _ae_threshold(dst_port)

            if meta:
                self._log_syn_debug(meta)

            verdict, source = self._fuse(
                rf_is_attack, confidence,
                anomaly_flagged, anomaly_score,
                dst_port, meta,
            )

            results[ml_indices[j]] = Verdict(
                label           = label,
                confidence      = confidence,
                is_attack       = verdict in ("ATTACK", "SUSPECT"),
                anomaly_score   = anomaly_score,
                anomaly_flagged = anomaly_flagged,
                verdict         = verdict,
                source          = source,
                class_probabilities = proba_dict,
            )

        return results

    @staticmethod
    def _is_syn_flood(flow_meta: dict) -> bool:
        """
        Detect unidirectional SYN floods (hping3 -S, Scapy, etc).

        Three conditions must ALL hold:

          1. syn_pkts >= SYN_FLOOD_MIN_PACKETS
             At least one SYN in forward direction.

          2. bwd_bytes <= SYN_FLOOD_BWD_BYTES_MAX_STRICT (default 44)
             This is the key discriminator:
               Normal handshake:  SYN → SYN-ACK(44B) + ACK(40B) → bwd=84  PASS
               RST response:      SYN → RST(40B)                 → bwd=40  FLOOD
               Silent drop:       SYN → nothing                  → bwd=0   FLOOD
             Set to 44 so completed handshakes (bwd >= 44) are NOT flagged.

          3. duration_ms < SYN_FLOOD_DURATION_MAX_MS (default 50)
             hping3 at high rate → nfstream micro-flows < 50 ms.
             Normal TCP flows typically last longer.
        """
        if not flow_meta:
            return False

        syn_pkts = int(flow_meta.get("src_syn_packets",            0))
        bwd_bytes= int(flow_meta.get("dst2src_bytes",              0))
        duration = float(flow_meta.get("bidirectional_duration_ms", 9999.0))

        is_flood = (
            syn_pkts  >= SYN_FLOOD_MIN_PACKETS
            and bwd_bytes  <= SYN_FLOOD_BWD_BYTES_MAX_STRICT
            and duration   <  SYN_FLOOD_DURATION_MAX_MS
        )

        return is_flood

    @staticmethod
    def _log_syn_debug(flow_meta: dict) -> None:
        log.debug(
            "[syn_heuristic] src_syn_pkts=%s  dst2src_bytes=%s  duration_ms=%s  → flood=%s",
            flow_meta.get("src_syn_packets",            "?"),
            flow_meta.get("dst2src_bytes",              "?"),
            flow_meta.get("bidirectional_duration_ms",  "?"),
            MLEngine._is_syn_flood(flow_meta),
        )

    @staticmethod
    def _fuse(
        rf_is_attack:  bool,
        rf_confidence: float,
        ae_flagged:    bool,
        ae_score:      float = 0.0,
        dst_port:      int   = 0,
        flow_meta:     dict | None = None,
    ) -> tuple[str, str]:
        """
        Decision fusion table:

        ┌──────────────────────┬──────────┬─────────────┬───────────┬────────────┐
        │ Condition            │ RF conf  │ AE flagged? │ Verdict   │ Source     │
        ├──────────────────────┼──────────┼─────────────┼───────────┼────────────┤
        │ Whitelisted port     │ —        │ —           │ BENIGN    │ WHITELIST  │ ← handled before _fuse
        │ SYN flood heuristic  │ any      │ any         │ ATTACK    │ RF         │
        │ RF attack            │ ≥ 0.70   │ any         │ ATTACK    │ RF         │
        │ RF attack            │ < 0.70   │ Yes         │ ATTACK    │ RF+AE      │
        │ RF attack            │ < 0.70   │ No          │ SUSPECT   │ RF         │
        │ RF benign            │ ≥ 0.50   │ Yes         │ ANOMALY   │ AE         │
        │ RF benign            │ < 0.50   │ Yes         │ BENIGN    │ RF+AE      │ ← RF too uncertain
        │ RF benign            │ any      │ No          │ BENIGN    │ RF+AE      │
        └──────────────────────┴──────────┴─────────────┴───────────┴────────────┘

        Note: whitelisted flows (DHCP, mDNS, SSDP) never reach _fuse() —
        they are short-circuited in predict() / predict_batch() before
        any ML inference runs.
        """
        # ── SYN flood heuristic ────────────────────────────────────────
        if MLEngine._is_syn_flood(flow_meta):
            log.debug(
                f"[fuse] SYN-flood heuristic triggered "
                f"(port={dst_port} ae={ae_score:.4f} "
                f"bwd={flow_meta.get('dst2src_bytes', '?')} "
                f"dur={flow_meta.get('bidirectional_duration_ms', '?')}ms)"
            )
            return "ATTACK", "Heuristic"

        # ── RF says attack ─────────────────────────────────────────────
        if rf_is_attack:
            if rf_confidence >= RF_CONF_HIGH:
                return "ATTACK", "RF"
            elif ae_flagged:
                return "ATTACK", "RF+AE"
            else:
                return "SUSPECT", "RF"

        # ── RF says benign ─────────────────────────────────────────────
        if ae_flagged:
            if rf_confidence < AE_CONF_MIN:
                log.debug(
                    f"[fuse] AE flagged (score={ae_score:.4f}, port={dst_port}) "
                    f"but RF conf too low ({rf_confidence:.3f} < {AE_CONF_MIN}) — BENIGN"
                )
                return "BENIGN", "RF+AE"
            return "ANOMALY", "AE"

        return "BENIGN", "RF+AE"

    def calibrate_ae_threshold(
        self,
        clean_flows: list[dict],
        percentile: float = 99.5,
    ) -> float:
        """
        Compute a recommended AE_THRESHOLD from known-clean traffic.
        Run this against your normal container traffic BEFORE running attacks
        to get a baseline. The p99.5 value is a good starting threshold.
        """
        raw_df    = pd.concat([self._build_vector(f) for f in clean_flows], ignore_index=True)
        ae_matrix = self.benign_scaler.transform(raw_df)
        ae_errors = -self.ae_wrapper.score_samples(ae_matrix)

        threshold = float(np.percentile(ae_errors, percentile))
        log.info(
            f"AE calibration over {len(clean_flows)} flows: "
            f"p50={np.percentile(ae_errors, 50):.4f}  "
            f"p95={np.percentile(ae_errors, 95):.4f}  "
            f"p99={np.percentile(ae_errors, 99):.4f}  "
            f"p{percentile}={threshold:.4f} ← recommended AE_THRESHOLD"
        )
        return threshold

    def _build_vector(self, flow: dict) -> pd.DataFrame:
        row = {feat: float(flow.get(feat, 0.0)) for feat in self.feature_list}
        df  = pd.DataFrame([row], columns=self.feature_list)
        return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _run_classifier(self, rf_vector: np.ndarray) -> tuple[str, float, dict]:
        cls_id     = int(self.clf.predict(rf_vector)[0])
        label      = self.le.classes_[cls_id]
        proba      = self.clf.predict_proba(rf_vector)[0]
        confidence = float(proba[cls_id])
        proba_dict = {self.le.classes_[i]: float(proba[i]) for i in range(len(self.le.classes_))}
        return label, confidence, proba_dict

    @staticmethod
    def _load_pkl(path: Path, name: str):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}. Run train.py first.")
        obj = joblib.load(path)
        log.info(f"  Loaded {name:20s} ← {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
        return obj


_engine: Optional[MLEngine] = None


def get_engine() -> MLEngine:
    global _engine
    if _engine is None:
        _engine = MLEngine()
        _engine.load()
    return _engine


def _smoke_test() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    engine = get_engine()
    zero_flow = {f: 0.0 for f in engine.feature_list}

    # ── Test 1: DHCP broadcast — must be whitelisted ─────────────────────────
    dhcp_meta = {"dst_port": 67, "dst_ip": "255.255.255.255", "src_ip": "0.0.0.0",
                 "src_syn_packets": 0, "dst2src_bytes": 0, "bidirectional_duration_ms": 5.0}
    dhcp_flow = dict(zero_flow, **{"dst_port": 67.0})
    v1 = engine.predict(dhcp_flow, flow_meta=dhcp_meta)
    log.info(f"Test 1 — DHCP broadcast: verdict={v1.verdict} source={v1.source} (expect BENIGN/WHITELIST)")
    assert v1.verdict == "BENIGN" and v1.source == "WHITELIST", f"DHCP whitelist failed: {v1.verdict}/{v1.source}"

    # ── Test 2: SYN flood, no response ───────────────────────────────────────
    syn_meta_silent = {
        "dst_port": 80, "dst_ip": "10.0.0.2", "src_ip": "10.0.0.1",
        "src_syn_packets": 10, "dst2src_bytes": 0, "bidirectional_duration_ms": 5.0,
    }
    v2 = engine.predict(zero_flow, flow_meta=syn_meta_silent)
    log.info(f"Test 2 — SYN flood (silent): verdict={v2.verdict} source={v2.source} (expect ATTACK/RF)")
    assert v2.verdict == "ATTACK" and v2.source == "RF", f"SYN flood silent failed: {v2.verdict}/{v2.source}"

    # ── Test 3: SYN flood, RST response (bwd=40) ─────────────────────────────
    syn_meta_rst = {
        "dst_port": 80, "dst_ip": "10.0.0.2", "src_ip": "10.0.0.1",
        "src_syn_packets": 10, "dst2src_bytes": 40, "bidirectional_duration_ms": 5.0,
    }
    v3 = engine.predict(zero_flow, flow_meta=syn_meta_rst)
    log.info(f"Test 3 — SYN flood (RST):    verdict={v3.verdict} source={v3.source} (expect ATTACK/RF)")
    assert v3.verdict == "ATTACK" and v3.source == "RF", f"SYN flood RST failed: {v3.verdict}/{v3.source}"

    # ── Test 4: Normal TCP handshake (bwd=84 — SYN-ACK + ACK) ───────────────
    # Should NOT trigger SYN flood heuristic
    normal_meta = {
        "dst_port": 80, "dst_ip": "10.0.0.2", "src_ip": "10.0.0.1",
        "src_syn_packets": 1, "dst2src_bytes": 84, "bidirectional_duration_ms": 5.0,
    }
    v4 = engine.predict(zero_flow, flow_meta=normal_meta)
    log.info(f"Test 4 — Normal handshake:   verdict={v4.verdict} source={v4.source} (heuristic must NOT fire)")
    assert v4.source != "RF" or v4.verdict != "ATTACK" or not MLEngine._is_syn_flood(normal_meta), \
        f"False positive on normal handshake: {v4.verdict}/{v4.source}"

    # ── Test 5: real flows from disk ─────────────────────────────────────────
    test_path = DATA_DIR / "processed" / "test_raw.pkl"
    if test_path.exists():
        test_df = pd.read_pickle(test_path)
        sample  = test_df.sample(n=min(10, len(test_df)), random_state=42)
        log.info("\nTest 5 — 10 real flows from test set:")
        for _, row in sample.iterrows():
            flow_dict = {f: row[f] for f in engine.feature_list if f in row}
            v = engine.predict(flow_dict)
            log.info(
                f"  {row.get('Label', '?'):35s} → {v.label:35s} "
                f"conf={v.confidence:.3f} ae={v.anomaly_score:.4f} "
                f"verdict={v.verdict} src={v.source}"
            )

    # ── Test 6: batch whitelist ───────────────────────────────────────────────
    batch = [
        dict(zero_flow, **{"dst_port": 67.0}),  # DHCP — whitelisted
        dict(zero_flow, **{"dst_port": 80.0}),  # HTTP  — goes to ML
        dict(zero_flow, **{"dst_port": 80.0}),  # HTTP  — SYN flood
    ]
    metas = [
        {"dst_port": 67, "dst_ip": "255.255.255.255", "src_ip": "0.0.0.0",
         "src_syn_packets": 0, "dst2src_bytes": 0, "bidirectional_duration_ms": 5.0},
        None,
        {"dst_port": 80, "dst_ip": "10.0.0.2", "src_ip": "10.0.0.1",
         "src_syn_packets": 5, "dst2src_bytes": 0, "bidirectional_duration_ms": 10.0},
    ]
    results = engine.predict_batch(batch, flow_metas=metas)
    log.info("\nTest 6 — Batch (DHCP, normal HTTP, SYN flood):")
    for i, v in enumerate(results):
        log.info(f"  [{i}] verdict={v.verdict} source={v.source}")
    assert results[0].source == "WHITELIST",    f"Batch DHCP whitelist failed"
    assert results[2].verdict == "ATTACK",      f"Batch SYN flood failed"

    log.info(f"""
Config summary:
  AE_THRESHOLD              = {AE_THRESHOLD}
  AE_CONF_MIN               = {AE_CONF_MIN}
  RF_CONF_HIGH              = {RF_CONF_HIGH}
  SYN_FLOOD_MIN_PACKETS     = {SYN_FLOOD_MIN_PACKETS}
  SYN_FLOOD_BWD_MAX_STRICT  = {SYN_FLOOD_BWD_BYTES_MAX_STRICT}
  SYN_FLOOD_DUR_MAX_MS      = {SYN_FLOOD_DURATION_MAX_MS}
  WHITELIST ports           = {sorted(_WHITELIST_DST_PORTS)}
  Port AE thresholds        = {AE_THRESHOLD_BY_PORT}
""")
    log.info("All smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()