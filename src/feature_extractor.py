"""
feature_extractor.py — nfstream flow → ML feature dict
========================================================
Converts a completed nfstream NFlow object into the exact 72-feature dict
that predict.py expects, matching the CIC-IDS2018 / CICFlowMeter column names
stored in feature_list.json.
"""

from __future__ import annotations
import math
from typing import Any


NFSTREAM_TO_CIC: dict[str, str] = {
    # --- Packet / byte counts ---
    "Tot Fwd Pkts"       : "src2dst_packets",
    "Tot Bwd Pkts"       : "dst2src_packets",
    "TotLen Fwd Pkts"    : "src2dst_bytes",
    "TotLen Bwd Pkts"    : "dst2src_bytes",

    # --- Packet length stats (forward) ---
    "Fwd Pkt Len Max"    : "src2dst_max_ps",
    "Fwd Pkt Len Min"    : "src2dst_min_ps",
    "Fwd Pkt Len Mean"   : "src2dst_mean_ps",
    "Fwd Pkt Len Std"    : "src2dst_stddev_ps",

    # --- Packet length stats (backward) ---
    "Bwd Pkt Len Max"    : "dst2src_max_ps",
    "Bwd Pkt Len Min"    : "dst2src_min_ps",
    "Bwd Pkt Len Mean"   : "dst2src_mean_ps",
    "Bwd Pkt Len Std"    : "dst2src_stddev_ps",

    # --- Inter-arrival times (flow level) ---
    "Flow IAT Mean"      : "bidirectional_mean_iat",
    "Flow IAT Std"       : "bidirectional_stddev_iat",
    "Flow IAT Max"       : "bidirectional_max_iat",
    "Flow IAT Min"       : "bidirectional_min_iat",

    # --- Inter-arrival times (forward) ---
    "Fwd IAT Tot"        : "src2dst_duration",
    "Fwd IAT Mean"       : "src2dst_mean_iat",
    "Fwd IAT Std"        : "src2dst_stddev_iat",
    "Fwd IAT Max"        : "src2dst_max_iat",
    "Fwd IAT Min"        : "src2dst_min_iat",

    # --- Inter-arrival times (backward) ---
    "Bwd IAT Tot"        : "dst2src_duration",
    "Bwd IAT Mean"       : "dst2src_mean_iat",
    "Bwd IAT Std"        : "dst2src_stddev_iat",
    "Bwd IAT Max"        : "dst2src_max_iat",
    "Bwd IAT Min"        : "dst2src_min_iat",

    # --- TCP flags (forward) ---
    "Fwd PSH Flags"      : "src2dst_psh_packets",
    "Fwd URG Flags"      : "src2dst_urg_packets",

    # --- TCP flags (backward) ---
    "Bwd PSH Flags"      : "dst2src_psh_packets",
    "Bwd URG Flags"      : "dst2src_urg_packets",

    # --- Header lengths ---
    "Fwd Header Len"     : "src2dst_header_bytes",
    "Bwd Header Len"     : "dst2src_header_bytes",

    # --- Packet length (combined) ---
    "Pkt Len Min"        : "bidirectional_min_ps",
    "Pkt Len Max"        : "bidirectional_max_ps",
    "Pkt Len Mean"       : "bidirectional_mean_ps",
    "Pkt Len Std"        : "bidirectional_stddev_ps",

    # --- TCP flag counts (bidirectional) ---
    "FIN Flag Cnt"       : "bidirectional_fin_packets",
    "SYN Flag Cnt"       : "bidirectional_syn_packets",
    "RST Flag Cnt"       : "bidirectional_rst_packets",
    "PSH Flag Cnt"       : "bidirectional_psh_packets",
    "ACK Flag Cnt"       : "bidirectional_ack_packets",
    "URG Flag Cnt"       : "bidirectional_urg_packets",
    "CWE Flag Count"     : "bidirectional_cwr_packets",
    "ECE Flag Cnt"       : "bidirectional_ece_packets",

    # --- Segment size averages ---
    "Pkt Size Avg"       : "bidirectional_mean_ps",
    "Fwd Seg Size Avg"   : "src2dst_mean_ps",
    "Bwd Seg Size Avg"   : "dst2src_mean_ps",

    # --- Subflow ---
    "Subflow Fwd Pkts"   : "src2dst_packets",
    "Subflow Fwd Byts"   : "src2dst_bytes",
    "Subflow Bwd Pkts"   : "dst2src_packets",
    "Subflow Bwd Byts"   : "dst2src_bytes",

    # --- Window sizes ---
    "Init Fwd Win Byts"  : "src2dst_init_win_bytes",
    "Init Bwd Win Byts"  : "dst2src_init_win_bytes",

    # --- Active / idle stats ---
    "Active Mean"        : "active_mean",
    "Active Std"         : "active_stddev",
    "Active Max"         : "active_max",
    "Active Min"         : "active_min",
    "Idle Mean"          : "idle_mean",
    "Idle Std"           : "idle_stddev",
    "Idle Max"           : "idle_max",
    "Idle Min"           : "idle_min",
}

COMPUTED_FEATURES = {
    "Dst Port", "Protocol", "Flow Duration",
    "Flow Byts/s", "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s",
    "Down/Up Ratio", "Pkt Len Var", "Fwd Act Data Pkts", "Fwd Seg Size Min",
}


def _safe(flow: Any, attr: str, default: float = 0.0) -> float:
    val = getattr(flow, attr, default)
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def flow_to_features(flow: Any) -> dict[str, float]:
    """
    Convert a completed nfstream NFlow object to the 72-feature dict
    expected by predict.py.
    """
    features: dict[str, float] = {}

    # 1. Direct attribute mappings
    for cic_name, nf_attr in NFSTREAM_TO_CIC.items():
        if cic_name not in COMPUTED_FEATURES:
            features[cic_name] = _safe(flow, nf_attr)

    # 2. Computed features
    features["Dst Port"] = _safe(flow, "dst_port")
    features["Protocol"] = _safe(flow, "protocol")

    # Flow duration: nfstream gives ms, CICFlowMeter uses µs
    duration_ms = _safe(flow, "bidirectional_duration_ms")
    features["Flow Duration"] = duration_ms * 1000.0

    duration_s = duration_ms / 1000.0
    if duration_s > 0:
        features["Flow Byts/s"] = _safe(flow, "bidirectional_bytes")   / duration_s
        features["Flow Pkts/s"] = _safe(flow, "bidirectional_packets") / duration_s
        features["Fwd Pkts/s"]  = _safe(flow, "src2dst_packets")       / duration_s
        features["Bwd Pkts/s"]  = _safe(flow, "dst2src_packets")       / duration_s
    else:
        features["Flow Byts/s"] = features["Flow Pkts/s"] = 0.0
        features["Fwd Pkts/s"]  = features["Bwd Pkts/s"]  = 0.0

    fwd = _safe(flow, "src2dst_packets")
    bwd = _safe(flow, "dst2src_packets")
    features["Down/Up Ratio"]    = (bwd / fwd) if fwd > 0 else 0.0
    features["Pkt Len Var"]      = _safe(flow, "bidirectional_stddev_ps") ** 2
    features["Fwd Act Data Pkts"]= _safe(flow, "src2dst_packets")
    features["Fwd Seg Size Min"] = _safe(flow, "src2dst_min_ps")

    # 3. Final safety pass
    for k, v in features.items():
        if math.isnan(v) or math.isinf(v):
            features[k] = 0.0

    return features


def flow_to_meta(flow: Any) -> dict[str, Any]:
    """
    Extract flow identity + heuristic fields needed by predict.py.

    The three heuristic fields at the bottom are consumed by
    MLEngine._is_syn_flood() — they MUST be present for SYN flood
    detection to work at runtime.
    """
    # nfstream >= 0.9.7 exposes per-direction TCP flag counts.
    # Try both known attribute name variants defensively.
    src_syn = getattr(flow, "src2dst_syn_packets", None)
    if src_syn is None:
        src_syn = getattr(flow, "bidirectional_syn_packets", None)
    if src_syn is None:
        src_syn = 0

    return {
        # --- Identity (not fed to ML model) ---
        "src_ip"       : getattr(flow, "src_ip",  ""),
        "dst_ip"       : getattr(flow, "dst_ip",  ""),
        "src_port"     : int(_safe(flow, "src_port")),
        "dst_port"     : int(_safe(flow, "dst_port")),
        "protocol"     : int(_safe(flow, "protocol")),
        "start_time_ms": int(_safe(flow, "bidirectional_first_seen_ms")),
        "end_time_ms"  : int(_safe(flow, "bidirectional_last_seen_ms")),

        # --- Heuristic fields (consumed by MLEngine._is_syn_flood) ---
        # src_syn_packets:           how many SYNs sent in forward direction
        # dst2src_bytes:             bytes received back from target
        # bidirectional_duration_ms: total flow duration
        "src_syn_packets"          : int(src_syn),
        "dst2src_bytes"            : int(_safe(flow, "dst2src_bytes")),
        "bidirectional_duration_ms": float(_safe(flow, "bidirectional_duration_ms")),
    }