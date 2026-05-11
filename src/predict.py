"""
predict.py — Runtime inference engine (RF + AE fusion + heuristic fallback)
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

AE_THRESHOLD:             float = float(os.getenv("AE_THRESHOLD",             "0.45"))
AE_SCORE_HARD_THRESHOLD:  float = float(os.getenv("AE_SCORE_HARD_THRESHOLD",  "5.0"))
RF_CONF_HIGH:             float = float(os.getenv("RF_CONF_HIGH",             "0.70"))
BENIGN_LABEL:             str   = os.getenv("BENIGN_LABEL", "Benign")

# AE_CONF_MIN removed — it was suppressing valid anomalies (e.g. Hydra FTP AE=14.82
# was silently dropped because RF conf < 0.50). All AE flags now surface.

AE_THRESHOLD_BY_PORT: dict[int, float] = {
    53:   float(os.getenv("AE_THRESHOLD_PORT_53",   "2.0")),
    67:   float(os.getenv("AE_THRESHOLD_PORT_67",   "999.0")),
    68:   float(os.getenv("AE_THRESHOLD_PORT_68",   "999.0")),
    80:   float(os.getenv("AE_THRESHOLD_PORT_80",   "0.55")),
    443:  float(os.getenv("AE_THRESHOLD_PORT_443",  "1.0")),
    1900: float(os.getenv("AE_THRESHOLD_PORT_1900", "999.0")),
}

# Protocol-level AE thresholds — checked when dst_port==0 (e.g. ICMP).
# ICMP (protocol 1): normal ping scores ae≈2.12; set threshold well above
# that so routine pings never trip the AE gate. ICMP floods are caught
# independently by _rule_icmp_flood in heuristics.py (pkt count ≥ 50),
# so raising this threshold does NOT create a blind spot for floods.
AE_THRESHOLD_BY_PROTOCOL: dict[int, float] = {
    1: float(os.getenv("AE_THRESHOLD_PROTO_ICMP", "3.5")),    # ICMP  — ping baseline ≈2.12
    2: float(os.getenv("AE_THRESHOLD_PROTO_IGMP", "999.0")),  # IGMP  — multicast, always benign
}

_WHITELIST_DST_PORTS: frozenset[int] = frozenset(
    int(p) for p in os.getenv("WHITELIST_DST_PORTS", "67,68,5353,1900,631").split(",")
    if p.strip()
)


def _ae_threshold(dst_port: int, protocol: int = 0) -> float:
    """Return the AE anomaly score threshold for this flow.

    Port-specific thresholds take priority. For port-less protocols (ICMP,
    IGMP — dst_port==0), fall back to the protocol-level table so that
    normal ICMP traffic is never mislabelled as ANOMALY.
    """
    if dst_port in AE_THRESHOLD_BY_PORT:
        return AE_THRESHOLD_BY_PORT[dst_port]
    if dst_port == 0 and protocol in AE_THRESHOLD_BY_PROTOCOL:
        return AE_THRESHOLD_BY_PROTOCOL[protocol]
    return AE_THRESHOLD


def _is_ipv6(addr: str) -> bool:
    """Return True if *addr* looks like an IPv6 address.

    Uses a colon-presence heuristic that avoids importing the `ipaddress`
    module on the hot path; a colon cannot appear in a valid IPv4 address.
    """
    return ":" in addr


def _is_whitelisted(dst_port: int, src_ip: str = "", dst_ip: str = "") -> bool:
    # IPv6 — model trained exclusively on IPv4; skip all v6 flows to avoid
    # spurious anomaly scores from the out-of-distribution address space.
    if _is_ipv6(src_ip) or _is_ipv6(dst_ip):
        return True
    if dst_port in _WHITELIST_DST_PORTS:
        return True
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
    heuristic_flags:     list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label"              : self.label,
            "confidence"         : round(self.confidence, 4),
            "is_attack"          : self.is_attack,
            "anomaly_score"      : round(self.anomaly_score, 6),
            "anomaly_flagged"    : self.anomaly_flagged,
            "verdict"            : self.verdict,
            "source"             : self.source,
            "heuristic_flags"    : self.heuristic_flags,
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

    def load(self) -> None:
        if self._loaded:
            return

        log.info("MLEngine: loading artefacts ...")
        self.clf           = self._load_pkl(MODELS_DIR / "classifier.pkl",    "RandomForest")
        self.ae_wrapper    = self._load_pkl(MODELS_DIR / "anomaly.pkl",       "AutoencoderWrapper")
        self.scaler        = self._load_pkl(MODELS_DIR / "scaler.pkl",        "scaler")
        self.benign_scaler = self._load_pkl(MODELS_DIR / "benign_scaler.pkl", "benign_scaler")
        self.le            = self._load_pkl(MODELS_DIR / "label_encoder.pkl", "LabelEncoder")

        with open(DATA_DIR / "feature_list.json") as f:
            self.feature_list = json.load(f)

        log.info(
            f"MLEngine: ready — {len(self.feature_list)} features, "
            f"{len(self.le.classes_)} classes | "
            f"AE_THRESHOLD={AE_THRESHOLD} | "
            f"AE_SCORE_HARD_THRESHOLD={AE_SCORE_HARD_THRESHOLD} | "
            f"AE_THRESHOLD_ICMP={AE_THRESHOLD_BY_PROTOCOL.get(1, 'N/A')} | "
            f"RF_CONF_HIGH={RF_CONF_HIGH} | "
            f"WHITELIST ports={sorted(_WHITELIST_DST_PORTS)}"
        )
        self._loaded = True

    def predict(self, flow: dict, flow_meta: dict | None = None) -> "Verdict":
        if not self._loaded:
            self.load()

        dst_port = int(flow.get("dst_port", 0))
        protocol = int((flow_meta or {}).get("protocol", flow.get("Protocol", 0)))
        dst_ip   = flow_meta.get("dst_ip", "") if flow_meta else ""
        src_ip   = flow_meta.get("src_ip", "") if flow_meta else ""

        if _is_whitelisted(dst_port, src_ip, dst_ip):
            log.debug(f"[predict] Whitelisted port={dst_port} dst={dst_ip}")
            return Verdict(
                label="Benign", confidence=1.0, is_attack=False,
                anomaly_score=0.0, anomaly_flagged=False,
                verdict="BENIGN", source="WHITELIST",
            )

        raw_df = self._build_vector(flow)

        rf_vector                         = self.scaler.transform(raw_df)
        rf_label, rf_confidence, rf_proba = self._run_classifier(rf_vector)
        rf_is_attack                      = (rf_label != BENIGN_LABEL)

        ae_vector       = self.benign_scaler.transform(raw_df)
        anomaly_score   = float(-self.ae_wrapper.score_samples(ae_vector)[0])
        anomaly_flagged = anomaly_score > _ae_threshold(dst_port, protocol)

        verdict, source = self._fuse(rf_is_attack, rf_confidence, anomaly_flagged, anomaly_score)

        # Heuristic layer — runs after ML fusion, can upgrade verdict only
        heuristic_flags: list[str] = []
        verdict, source, heuristic_flags = apply_heuristics(
            flow=flow,
            meta=flow_meta or {},
            verdict=verdict,
            source=source,
            flags=heuristic_flags,
        )

        return Verdict(
            label               = rf_label,
            confidence          = rf_confidence,
            is_attack           = verdict in ("ATTACK", "SUSPECT"),
            anomaly_score       = anomaly_score,
            anomaly_flagged     = anomaly_flagged,
            verdict             = verdict,
            source              = source,
            class_probabilities = rf_proba,
            heuristic_flags     = heuristic_flags,
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

        results    = [None] * len(flows)
        ml_indices = []
        ml_flows   = []
        ml_metas   = []

        for i, (flow, meta) in enumerate(zip(flows, flow_metas)):
            dst_port = int(flow.get("dst_port", 0))
            dst_ip   = meta.get("dst_ip", "") if meta else ""
            src_ip   = meta.get("src_ip", "") if meta else ""

            if _is_whitelisted(dst_port, src_ip, dst_ip):
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
            protocol        = int((meta or {}).get("protocol", flow.get("Protocol", 0)))
            anomaly_score   = float(ae_errors[j])
            anomaly_flagged = anomaly_score > _ae_threshold(dst_port, protocol)

            verdict, source = self._fuse(rf_is_attack, confidence, anomaly_flagged, anomaly_score)

            # Heuristic layer
            heuristic_flags: list[str] = []
            verdict, source, heuristic_flags = apply_heuristics(
                flow=flow,
                meta=meta or {},
                verdict=verdict,
                source=source,
                flags=heuristic_flags,
            )

            results[ml_indices[j]] = Verdict(
                label               = label,
                confidence          = confidence,
                is_attack           = verdict in ("ATTACK", "SUSPECT"),
                anomaly_score       = anomaly_score,
                anomaly_flagged     = anomaly_flagged,
                verdict             = verdict,
                source              = source,
                class_probabilities = proba_dict,
                heuristic_flags     = heuristic_flags,
            )

        return results

    @staticmethod
    def _fuse(
        rf_is_attack:  bool,
        rf_confidence: float,
        ae_flagged:    bool,
        ae_score:      float = 0.0,
    ) -> tuple[str, str]:
        """
        ML decision fusion — RF classifier + Autoencoder.

        Changes from v1:
          • AE_SCORE_HARD_THRESHOLD: extreme AE score (e.g. Hydra FTP = 14.82) always
            fires ATTACK regardless of RF confidence. This was the single biggest miss.
          • AE_CONF_MIN removed: the branch that silently dropped AE anomalies when
            RF confidence was low is gone. All AE flags now surface as ANOMALY.

        ┌──────────────────────────┬──────────┬─────────────┬───────────┬────────────┐
        │ Condition                │ RF conf  │ AE flagged? │ Verdict   │ Source     │
        ├──────────────────────────┼──────────┼─────────────┼───────────┼────────────┤
        │ AE score ≥ hard thresh   │ any      │ any         │ ATTACK    │ AE_HARD    │
        │ RF attack                │ ≥ 0.70   │ any         │ ATTACK    │ RF         │
        │ RF attack                │ < 0.70   │ Yes         │ ATTACK    │ RF+AE      │
        │ RF attack                │ < 0.70   │ No          │ SUSPECT   │ RF         │
        │ RF benign                │ any      │ Yes         │ ANOMALY   │ AE         │
        │ RF benign                │ any      │ No          │ BENIGN    │ RF+AE      │
        └──────────────────────────┴──────────┴─────────────┴───────────┴────────────┘
        """
        # Hard override: catastrophic anomaly score — trust AE unconditionally
        if ae_score >= AE_SCORE_HARD_THRESHOLD:
            return "ATTACK", "AE_HARD"

        if rf_is_attack:
            if rf_confidence >= RF_CONF_HIGH:
                return "ATTACK", "RF"
            elif ae_flagged:
                return "ATTACK", "RF+AE"
            else:
                return "SUSPECT", "RF"

        # RF says benign — surface any AE anomaly (AE_CONF_MIN gate removed)
        if ae_flagged:
            return "ANOMALY", "AE"

        return "BENIGN", "RF+AE"

    def calibrate_ae_threshold(self, clean_flows: list[dict], percentile: float = 99.5) -> float:
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


# ---------------------------------------------------------------------------
# Heuristic layer import (keep at bottom to avoid circular imports)
# ---------------------------------------------------------------------------
from heuristics import apply_heuristics   # noqa: E402


def _smoke_test() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    engine    = get_engine()
    zero_flow = {f: 0.0 for f in engine.feature_list}

    # Test 1: DHCP whitelist
    dhcp_flow = dict(zero_flow, **{"dst_port": 67.0})
    dhcp_meta = {"dst_port": 67, "dst_ip": "255.255.255.255", "src_ip": "0.0.0.0"}
    v1 = engine.predict(dhcp_flow, flow_meta=dhcp_meta)
    log.info(f"Test 1 — DHCP: verdict={v1.verdict} source={v1.source} (expect BENIGN/WHITELIST)")
    assert v1.verdict == "BENIGN" and v1.source == "WHITELIST"

    # Test 2: Simulated Hydra FTP — high AE score should trigger AE_HARD
    hydra_flow = dict(zero_flow, **{"dst_port": 21.0, "bidirectional_syn_packets": 50.0})
    hydra_meta = {"dst_port": 21, "dst_ip": "10.0.0.1", "src_ip": "192.168.1.100"}
    v2 = engine.predict(hydra_flow, flow_meta=hydra_meta)
    log.info(f"Test 2 — Hydra-like: verdict={v2.verdict} ae={v2.anomaly_score:.4f} flags={v2.heuristic_flags}")

    # Test 3: zero flow goes through ML
    v3 = engine.predict(zero_flow)
    log.info(f"Test 3 — Zero flow: verdict={v3.verdict} label={v3.label} conf={v3.confidence:.4f}")

    # Test 4: IPv6 whitelist
    ipv6_flow = dict(zero_flow, **{"dst_port": 443.0})
    ipv6_meta = {"dst_port": 443, "dst_ip": "2001:db8::1", "src_ip": "fe80::1"}
    v4 = engine.predict(ipv6_flow, flow_meta=ipv6_meta)
    log.info(f"Test 4 — IPv6: verdict={v4.verdict} source={v4.source} (expect BENIGN/WHITELIST)")
    assert v4.verdict == "BENIGN" and v4.source == "WHITELIST"

    # Test 5: real flows from disk
    test_path = DATA_DIR / "processed" / "test_raw.pkl"
    if test_path.exists():
        test_df = pd.read_pickle(test_path)
        sample  = test_df.sample(n=min(10, len(test_df)), random_state=42)
        log.info("\nTest 5 — 10 real flows:")
        for _, row in sample.iterrows():
            flow_dict = {f: row[f] for f in engine.feature_list if f in row}
            v = engine.predict(flow_dict)
            log.info(
                f"  {row.get('Label', '?'):35s} → {v.label:35s} "
                f"conf={v.confidence:.3f} ae={v.anomaly_score:.4f} "
                f"verdict={v.verdict} src={v.source} heur={v.heuristic_flags}"
            )

    log.info(
        f"\nConfig: AE_THRESHOLD={AE_THRESHOLD}  "
        f"AE_SCORE_HARD_THRESHOLD={AE_SCORE_HARD_THRESHOLD}  "
        f"RF_CONF_HIGH={RF_CONF_HIGH}  "
        f"WHITELIST={sorted(_WHITELIST_DST_PORTS)}"
    )
    log.info("Smoke test done.")


if __name__ == "__main__":
    _smoke_test()