"""
feature_extractor.py — nfstream flow → ML feature dict
========================================================
Converts a completed nfstream NFlow object into the exact 72-feature dict
that predict.py expects, matching the CIC-IDS2018 / CICFlowMeter column names
stored in feature_list.json.

nfstream computes CICFlowMeter-compatible statistics natively — we just need
to map nfstream attribute names to the exact column names used during training.

Usage (called by capture.py):
    from feature_extractor import flow_to_features
    features = flow_to_features(flow)   # flow is an nfstream NFlow object
    verdict  = engine.predict(features)
"""

from __future__ import annotations
import math
from typing import Any


# ---------------------------------------------------------------------------
# nfstream attribute → CICFlowMeter column name mapping
# ---------------------------------------------------------------------------
# Each entry: "cic_column_name": "nfstream_attribute_name"
# If the attribute doesn't exist on the NFlow object, the feature defaults to 0.
#
# Reference:
#   nfstream docs:     https://www.nfstream.org/docs/api
#   CICFlowMeter cols: data/feature_list.json (72 features after preprocessing)
#
# to ask: if nfstream version changes or attribute names differ,
# update the right-hand side values here. The left-hand side (CIC names)
# must never change — they're locked to the trained model.

NFSTREAM_TO_CIC: dict[str, str] = {
    # --- Basic flow identifiers (kept for alert.py, not fed to ML model) ---
    # These are extracted separately in flow_to_meta(), not in flow_to_features()

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

    # --- Flow-level rates ---
    "Flow Byts/s"        : "bidirectional_bytes",   # computed below, not direct
    "Flow Pkts/s"        : "bidirectional_packets",  # computed below, not direct

    # --- Inter-arrival times (flow level) ---
    "Flow IAT Mean"      : "bidirectional_mean_iat",
    "Flow IAT Std"       : "bidirectional_stddev_iat",
    "Flow IAT Max"       : "bidirectional_max_iat",
    "Flow IAT Min"       : "bidirectional_min_iat",

    # --- Inter-arrival times (forward) ---
    "Fwd IAT Tot"        : "src2dst_duration",       # proxy — see note below
    "Fwd IAT Mean"       : "src2dst_mean_iat",
    "Fwd IAT Std"        : "src2dst_stddev_iat",
    "Fwd IAT Max"        : "src2dst_max_iat",
    "Fwd IAT Min"        : "src2dst_min_iat",

    # --- Inter-arrival times (backward) ---
    "Bwd IAT Tot"        : "dst2src_duration",       # proxy
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

    # --- Packet rates (directional) ---
    "Fwd Pkts/s"         : "src2dst_packets",        # computed below
    "Bwd Pkts/s"         : "dst2src_packets",        # computed below

    # --- Packet length (combined) ---
    "Pkt Len Min"        : "bidirectional_min_ps",
    "Pkt Len Max"        : "bidirectional_max_ps",
    "Pkt Len Mean"       : "bidirectional_mean_ps",
    "Pkt Len Std"        : "bidirectional_stddev_ps",
    "Pkt Len Var"        : "bidirectional_stddev_ps", # squared below

    # --- TCP flag counts (bidirectional) ---
    "FIN Flag Cnt"       : "bidirectional_fin_packets",
    "SYN Flag Cnt"       : "bidirectional_syn_packets",
    "RST Flag Cnt"       : "bidirectional_rst_packets",
    "PSH Flag Cnt"       : "bidirectional_psh_packets",
    "ACK Flag Cnt"       : "bidirectional_ack_packets",
    "URG Flag Cnt"       : "bidirectional_urg_packets",
    "CWE Flag Count"     : "bidirectional_cwr_packets",
    "ECE Flag Cnt"       : "bidirectional_ece_packets",

    # --- Ratios ---
    "Down/Up Ratio"      : "dst2src_packets",        # computed below

    # --- Segment size averages ---
    "Pkt Size Avg"       : "bidirectional_mean_ps",
    "Fwd Seg Size Avg"   : "src2dst_mean_ps",
    "Bwd Seg Size Avg"   : "dst2src_mean_ps",

    # --- Subflow (nfstream = same as full flow for single subflow) ---
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

# Features that need special computation (not a direct attribute lookup)
# handled explicitly in flow_to_features()
COMPUTED_FEATURES = {
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Down/Up Ratio",
    "Pkt Len Var",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
}


def _safe(flow: Any, attr: str, default: float = 0.0) -> float:
    """
    Safely read a float attribute from an nfstream NFlow.
    Returns default if attribute is missing, None, NaN, or Inf.
    """
    val = getattr(flow, attr, default)
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def flow_to_features(flow: Any) -> dict[str, float]:
    """
    Convert a completed nfstream NFlow object to the 72-feature dict
    expected by predict.py.

    Args:
        flow: nfstream NFlow object (from NFStreamer callback or iteration)

    Returns:
        dict mapping CICFlowMeter column name → float value
        Missing or invalid values default to 0.0.
    """
    features: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Direct attribute mappings
    # ------------------------------------------------------------------
    for cic_name, nf_attr in NFSTREAM_TO_CIC.items():
        if cic_name not in COMPUTED_FEATURES:
            features[cic_name] = _safe(flow, nf_attr)

    # ------------------------------------------------------------------
    # 2. Computed features
    # ------------------------------------------------------------------

    # Destination port and protocol
    features["Dst Port"] = _safe(flow, "dst_port")
    features["Protocol"] = _safe(flow, "protocol")

    # Flow duration in microseconds (nfstream gives milliseconds)
    # CICFlowMeter uses microseconds
    duration_ms = _safe(flow, "bidirectional_duration_ms")
    duration_us = duration_ms * 1000.0
    features["Flow Duration"] = duration_us

    # Flow Bytes/s and Packets/s
    # Guard against zero-duration flows (would give Inf)
    duration_s = duration_ms / 1000.0
    if duration_s > 0:
        total_bytes   = _safe(flow, "bidirectional_bytes")
        total_packets = _safe(flow, "bidirectional_packets")
        fwd_packets   = _safe(flow, "src2dst_packets")
        bwd_packets   = _safe(flow, "dst2src_packets")

        features["Flow Byts/s"] = total_bytes   / duration_s
        features["Flow Pkts/s"] = total_packets / duration_s
        features["Fwd Pkts/s"]  = fwd_packets   / duration_s
        features["Bwd Pkts/s"]  = bwd_packets   / duration_s
    else:
        features["Flow Byts/s"] = 0.0
        features["Flow Pkts/s"] = 0.0
        features["Fwd Pkts/s"]  = 0.0
        features["Bwd Pkts/s"]  = 0.0

    # Down/Up ratio = bwd_packets / fwd_packets
    fwd_pkts = _safe(flow, "src2dst_packets")
    bwd_pkts = _safe(flow, "dst2src_packets")
    features["Down/Up Ratio"] = (bwd_pkts / fwd_pkts) if fwd_pkts > 0 else 0.0

    # Packet length variance = stddev²
    stddev = _safe(flow, "bidirectional_stddev_ps")
    features["Pkt Len Var"] = stddev ** 2

    # Fwd Act Data Pkts — packets carrying actual data (non-zero payload)
    # nfstream doesn't expose this directly; use fwd packets as proxy
    # NOTE: if nfstream adds this attribute, replace with:
    # features["Fwd Act Data Pkts"] = _safe(flow, "src2dst_data_packets")
    features["Fwd Act Data Pkts"] = _safe(flow, "src2dst_packets")

    # Fwd Seg Size Min — minimum segment size in forward direction
    # nfstream doesn't expose this directly; use min packet size as proxy
    # NOTE: replace with actual attribute if available in your nfstream version
    features["Fwd Seg Size Min"] = _safe(flow, "src2dst_min_ps")

    # ------------------------------------------------------------------
    # 3. Final safety pass — replace any remaining NaN/Inf with 0
    # ------------------------------------------------------------------
    for k, v in features.items():
        if math.isnan(v) or math.isinf(v):
            features[k] = 0.0

    return features


def flow_to_meta(flow: Any) -> dict[str, Any]:
    """
    Extract flow identity fields (not ML features) needed by alert.py
    to tell the Mitigation Engine which specific flow to block.

    These fields are NOT passed to the ML model — they were dropped
    during preprocessing (IDENTITY_COLS in preprocess.py).

    Returns:
        dict with src/dst IP, ports, protocol, timestamps
    """
    return {
        "src_ip"       : getattr(flow, "src_ip",   ""),
        "dst_ip"       : getattr(flow, "dst_ip",   ""),
        "src_port"     : int(_safe(flow, "src_port")),
        "dst_port"     : int(_safe(flow, "dst_port")),
        "protocol"     : int(_safe(flow, "protocol")),
        "start_time_ms": int(_safe(flow, "bidirectional_first_seen_ms")),
        "end_time_ms"  : int(_safe(flow, "bidirectional_last_seen_ms")),
    }