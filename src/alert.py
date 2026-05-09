"""
alert.py — POST verdict to the Mitigation Engine
==================================================
Called by capture.py for every non-BENIGN verdict.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.config import Config
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP session with retry logic
# ---------------------------------------------------------------------------
_session: requests.Session | None = None

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        retry = Retry(
            total            = Config.ALERT_RETRIES,
            backoff_factor   = 0.3,
            status_forcelist = [500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        _session.mount("http://",  adapter)
        _session.mount("https://", adapter)
    return _session


# ---------------------------------------------------------------------------
# Attack type resolver
# ---------------------------------------------------------------------------
def _resolve_attack_type(
    verdict_dict: dict[str, Any],
    meta: dict[str, Any],
) -> str:
    """
    Return a human-readable attack type for the dashboard TYPE D'ATTAQUE column.

    Priority:
      1. If RF label is meaningful (not Benign) → use it directly
      2. If source is HEURISTIC               → SYN Flood
      3. If verdict is ANOMALY                → infer from port/flags
      4. Fallback                             → Unknown
    """
    rf_label = verdict_dict.get("label", "")
    source   = verdict_dict.get("source", "")
    verdict  = verdict_dict.get("verdict", "")
    dst_port = int(meta.get("dst_port", 0))
    src_port = int(meta.get("src_port", 0))
    syn_pkts = int(meta.get("src_syn_packets", 0))

    # RF gave a real attack label — trust it
    if rf_label and rf_label != "Benign":
        return rf_label

    
    if source == "HEURISTIC":
        return "SYN Flood"

    
    if verdict in ("ANOMALY", "ATTACK", "SUSPECT"):
        # Port scan: many different dst ports, one SYN each
        if syn_pkts == 1:
            if dst_port == 22:
                return "SSH Scan"
            if dst_port == 21:
                return "FTP Scan"
            if dst_port in (80, 8080, 8000, 8443):
                return "HTTP Scan"
            if dst_port == 443:
                return "HTTPS Scan"
            if dst_port == 3389:
                return "RDP Scan"
            return "Port Scan"

        # High SYN count → flood
        if syn_pkts > 1:
            return "SYN Flood"

        # Port-based inference for non-SYN flows
        if dst_port == 22 or src_port == 22:
            return "SSH Brute Force"
        if dst_port == 21 or src_port == 21:
            return "FTP Brute Force"
        if dst_port in (80, 8080, 8000):
            return "HTTP DoS"
        if dst_port == 443:
            return "HTTPS DoS"

        return "Anomalous Flow"

    return "Unknown"


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------
def _build_payload(
    verdict_dict: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
   
    attack_type = _resolve_attack_type(verdict_dict, meta)

    return {
        # Core decision
        "action"         : verdict_dict.get("action",    "pass"),
        "verdict"        : verdict_dict.get("verdict",   "BENIGN"),
        "label"          : attack_type,          # ← human-readable, replaces raw RF label
        "rf_label"       : verdict_dict.get("label", ""),   # ← original RF label kept for debugging
        "confidence"     : verdict_dict.get("confidence", 0.0),
        "source"         : verdict_dict.get("source",    ""),

        # Flow identity
        "src_ip"         : meta.get("src_ip",    ""),
        "dst_ip"         : meta.get("dst_ip",    ""),
        "src_port"       : meta.get("src_port",   0),
        "dst_port"       : meta.get("dst_port",   0),
        "protocol"       : meta.get("protocol",   0),

        # Timestamps
        "start_time_ms"  : meta.get("start_time_ms", 0),
        "end_time_ms"    : meta.get("end_time_ms",   0),

        # AE enrichment
        "anomaly_score"  : verdict_dict.get("anomaly_score",   None),
        "anomaly_flagged": verdict_dict.get("anomaly_flagged", False),
    }


# ---------------------------------------------------------------------------
# Main send function
# ---------------------------------------------------------------------------
def send_alert(
    verdict_dict: dict[str, Any],
    meta: dict[str, Any],
) -> bool:
    """
    POST a verdict to the Mitigation Engine.

    Args:
        verdict_dict: output of Verdict.to_dict() from predict.py
        meta:         output of flow_to_meta() from feature_extractor.py

    Returns:
        True if delivered successfully, False otherwise.
    """
    verdict = verdict_dict.get("verdict", "BENIGN")

    if verdict == "BENIGN":
        return True

    payload = _build_payload(verdict_dict, meta)
    session = _get_session()

    # Log what we're about to send
    log.info(
        f"[alert→] {verdict:7s}  "
        f"{payload['src_ip']} → {payload['dst_ip']}:{payload['dst_port']}  "
        f"label={payload['label']}  conf={payload['confidence']:.3f}"
    )

    t0 = time.perf_counter()
    try:
        response = session.post(
            Config.MITIGATION_URL,
            json=payload,
            timeout=Config.ALERT_TIMEOUT_S,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": Config.MITIGATION_API_KEY,
            },
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if response.ok:
            log.info(
                f"[alert] ✓ {payload['label']:30s}  "
                f"http={response.status_code}  ms={elapsed_ms:.1f}"
            )
            return True
        else:
            log.warning(
                f"[alert] Mitigation Engine returned {response.status_code}: "
                f"{response.text[:200]}"
            )
            return False

    except requests.exceptions.ConnectionError:
        log.error(
            f"[alert] Cannot reach Mitigation Engine at {Config.MITIGATION_URL} — "
            f"is it running? verdict={verdict} was NOT delivered."
        )
        return False

    except requests.exceptions.Timeout:
        log.error(
            f"[alert] Mitigation Engine timed out after {Config.ALERT_TIMEOUT_S}s"
        )
        return False

    except Exception as e:
        log.error(f"[alert] Unexpected error: {e}")
        return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    cases = [
        # (description, verdict_dict, meta, expected_label)
        (
            "nmap SYN scan → Port Scan",
            {"verdict": "ANOMALY", "label": "Benign", "source": "AE",
             "confidence": 0.52, "anomaly_score": 0.72, "anomaly_flagged": True},
            {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 59046,
             "dst_port": 80, "protocol": 6, "src_syn_packets": 1,
             "start_time_ms": 0, "end_time_ms": 10},
            "HTTP Scan",
        ),
        (
            "hping3 SYN flood → SYN Flood",
            {"verdict": "ATTACK", "label": "Benign", "source": "HEURISTIC",
             "confidence": 0.60, "anomaly_score": 0.41, "anomaly_flagged": False},
            {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 12345,
             "dst_port": 80, "protocol": 6, "src_syn_packets": 500,
             "start_time_ms": 0, "end_time_ms": 10},
            "SYN Flood",
        ),
        (
            "RF label present → use RF label",
            {"verdict": "ATTACK", "label": "FTP-BruteForce", "source": "RF",
             "confidence": 0.95, "anomaly_score": 0.80, "anomaly_flagged": True},
            {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 54321,
             "dst_port": 21, "protocol": 6, "src_syn_packets": 1,
             "start_time_ms": 0, "end_time_ms": 500},
            "FTP-BruteForce",
        ),
    ]

    all_passed = True
    for desc, vdict, meta, expected in cases:
        attack_type = _resolve_attack_type(vdict, meta)
        status = "✓" if attack_type == expected else "✗"
        log.info(f"  {status} {desc}: got='{attack_type}' expected='{expected}'")
        if attack_type != expected:
            all_passed = False

    if all_passed:
        log.info("All label resolution tests passed.")
    else:
        log.error("Some tests failed — check _resolve_attack_type()")


if __name__ == "__main__":
    _smoke_test()