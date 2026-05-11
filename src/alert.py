"""
alert.py — POST verdict to the Mitigation Engine
"""
from __future__ import annotations

import logging
import time
from typing import Any

from src.config import Config
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

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


 
_HEUR_LABEL_MAP: dict[str, str] = {
    "_rule_brute_force"    : "Brute Force",
    "_rule_slowloris"      : "DoS attacks-Slowloris",
    "_rule_syn_flood"      : "SYN Flood",
    "_rule_portscan"       : "Port Scan",
    "_rule_icmp_flood"     : "ICMP Flood",
    "_rule_empty_udp_probe": "UDP Probe",
    "_rule_web_scanner"    : "Web Scanner",
    "_rule_injection_tool" : "Injection Attack",
    "_rule_http_probe"     : "HTTP Probe",
}


def _resolve_label(verdict_dict: dict[str, Any]) -> str:
    """
    Pick the best human-readable label for the alert.

    Priority:
      1. Heuristic fired → always use heuristic label (most specific)
      2. RF gave a real attack label → use it
      3. Fallback → raw RF label
    """
    rf_label        = verdict_dict.get("label", "Benign")
    heuristic_flags = verdict_dict.get("heuristic_flags", [])

    # Heuristics fired — always prefer their label over RF
    for flag in heuristic_flags:
        rule_name = flag.split(":")[0]
        if rule_name in _HEUR_LABEL_MAP:
            return _HEUR_LABEL_MAP[rule_name]

    # No heuristic matched — fall back to RF label
    if rf_label and rf_label != "Benign":
        return rf_label

    return rf_label
def _build_payload(
    verdict_dict: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    label = _resolve_label(verdict_dict)

 
    raw_source = verdict_dict.get("source", "")
    ml_source  = raw_source.replace("HEUR:_rule_", "HEUR:").replace("_rule_", "")[:16]

    return {
        "action"          : verdict_dict.get("action",    "pass"),
        "verdict"         : verdict_dict.get("verdict",   "BENIGN"),
        "label"           : label,
        "rf_label"        : verdict_dict.get("label",     "Benign"),
        "confidence"      : verdict_dict.get("confidence", 0.0),
        "source"          : ml_source,                               # ← fixed
        "heuristic_flags" : verdict_dict.get("heuristic_flags", []),
        "src_ip"          : meta.get("src_ip",    ""),
        "dst_ip"          : meta.get("dst_ip",    ""),
        "src_port"        : meta.get("src_port",   0),
        "dst_port"        : meta.get("dst_port",   0),
        "protocol"        : meta.get("protocol",   0),
        "start_time_ms"   : meta.get("start_time_ms", 0),
        "end_time_ms"     : meta.get("end_time_ms",   0),
        "anomaly_score"   : verdict_dict.get("anomaly_score",   None),
        "anomaly_flagged" : verdict_dict.get("anomaly_flagged", False),
    }
def send_alert(
    verdict_dict: dict[str, Any],
    meta: dict[str, Any],
) -> bool:
    verdict = verdict_dict.get("verdict", "BENIGN")
    if verdict == "BENIGN":
        return True

    payload    = _build_payload(verdict_dict, meta)
    session    = _get_session()
    heur_flags = payload.get("heuristic_flags", [])
    heur_str   = f"  heur=[{','.join(f.split('(')[0] for f in heur_flags)}]" if heur_flags else ""

    t0 = time.perf_counter()
    try:
        response = session.post(
            Config.MITIGATION_URL,
            json=payload,
            timeout=Config.ALERT_TIMEOUT_S,
            headers={
                "Content-Type": "application/json",
                "X-API-Key"   : Config.MITIGATION_API_KEY,
            },
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if response.ok:
            log.info(
                f"[alert] ✓ {payload['label']:30s}  "
                f"http={response.status_code}  ms={elapsed_ms:.1f}"
                f"{heur_str}"
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
            f"verdict={verdict} was NOT delivered."
        )
        return False
    except requests.exceptions.Timeout:
        log.error(f"[alert] Mitigation Engine timed out after {Config.ALERT_TIMEOUT_S}s")
        return False
    except Exception as e:
        log.error(f"[alert] Unexpected error: {e}")
        return False