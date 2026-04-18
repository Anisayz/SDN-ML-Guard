"""
alert.py — POST verdict to the Mitigation Engine
==================================================
Called by capture.py for every non-BENIGN verdict.

The Mitigation Engine receives a JSON payload
and takes the appropriate action in OVS:
    ATTACK  → install drop rule
    SUSPECT → log alert, no OVS rule
    ANOMALY → install rate-limit meter
    BENIGN  → never called (capture.py filters these out)

Configuration (set in .env):
    MITIGATION_URL   — full URL of the Mitigation Engine endpoint
                       default: http://localhost:9000/alert
    ALERT_TIMEOUT_S  — HTTP timeout in seconds (default: 2.0)
    ALERT_RETRIES    — number of retries on failure (default: 2)

to ask anis
    This file sends a POST request to MITIGATION_URL with the JSON body
    described in _build_payload(). Adjust the payload fields to match
    whatever your engine expects.
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
            total            = ALERT_RETRIES,
            backoff_factor   = 0.3,
            status_forcelist = [500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        _session.mount("http://",  adapter)
        _session.mount("https://", adapter)
    return _session


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------
def _build_payload(
    verdict_dict: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the JSON payload sent to the Mitigation Engine.

    Current structure — adjust to match your anis api

    {
        "action":     "block" | "alert" | "rate_limit+alert" | "pass",
        "verdict":    "ATTACK" | "SUSPECT" | "ANOMALY" | "BENIGN",
        "label":      "DDoS attacks-LOIC-HTTP",   ← attack type from RF
        "confidence": 0.999,                       ← RF confidence
        "source":     "RF+AE",                     ← which model(s) triggered

        // Flow identity — used by OVS to match the specific flow to block
        "src_ip":     "192.168.1.10",
        "dst_ip":     "192.168.1.1",
        "src_port":   54321,
        "dst_port":   80,
        "protocol":   6,                           ← 6=TCP, 17=UDP

        // Timestamps (milliseconds since epoch)
        "start_time_ms": 1700000000000,
        "end_time_ms":   1700000001500,

        // Optional enrichment
        "anomaly_score":   -0.045,                 ← AE reconstruction error
        "anomaly_flagged": true,
    }

    to ask daghen if the engine expects a different structure,
    i should modify this function. Examples:
      - Remove fields you don't need
      - Rename keys (e.g. "action" → "mitigation_action")
      - Add fields (e.g. "switch_id", "in_port" from OVS metadata)
      - Change "protocol" from int to string ("TCP"/"UDP")
    """
    return {
        # Core decision
        "action"         : verdict_dict.get("action",    "pass"),
        "verdict"        : verdict_dict.get("verdict",   "BENIGN"),
        "label"          : verdict_dict.get("label",     "Unknown"),
        "confidence"     : verdict_dict.get("confidence", 0.0),
        "source"         : verdict_dict.get("source",    ""),

        # Flow identity — for OVS rule installation
        # to ask, these fields are critical for matching the flow in OVS
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
                      contains src/dst IP, ports, timestamps

    Returns:
        True if the alert was delivered successfully, False otherwise.
        Failures are logged but never raise — a logging failure must not
        crash the capture pipeline.
    """
    verdict = verdict_dict.get("verdict", "BENIGN")

    # Never send alerts for clean traffic
    if verdict == "BENIGN":
        return True

    payload = _build_payload(verdict_dict, meta)
    session = _get_session()

    t0 = time.perf_counter()
    try:
        response = session.post(
            Config.MITIGATION_URL,
            json=payload,
            timeout=Config.ALERT_TIMEOUT_S,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": Config.MITIGATIN_API_KEY,    
            },
        )
         
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if response.ok:
            log.info(
                f"[alert] {verdict:7s}  {payload['label']:30s}  "
                f"src={payload['src_ip']}:{payload['src_port']} → "
                f"dst={payload['dst_ip']}:{payload['dst_port']}  "
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
            f"[alert] Cannot reach Mitigation Engine at {MITIGATION_URL} — "
            f"is it running? verdict={verdict} was NOT delivered."
        )
        return False

    except requests.exceptions.Timeout:
        log.error(
            f"[alert] Mitigation Engine timed out after {ALERT_TIMEOUT_S}s — "
            f"verdict={verdict} may not have been delivered."
        )
        return False

    except Exception as e:
        log.error(f"[alert] Unexpected error sending alert: {e}")
        return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    test_verdict = {
        "verdict"        : "ATTACK",
        "action"         : "block",
        "label"          : "DDoS attacks-LOIC-HTTP",
        "confidence"     : 0.999,
        "source"         : "RF+AE",
        "anomaly_score"  : 0.045,
        "anomaly_flagged": True,
    }
    test_meta = {
        "src_ip"       : "10.0.0.1",
        "dst_ip"       : "10.0.0.2",
        "src_port"     : 54321,
        "dst_port"     : 80,
        "protocol"     : 6,
        "start_time_ms": int(time.time() * 1000) - 1500,
        "end_time_ms"  : int(time.time() * 1000),
    }

    log.info(f"Sending test alert to {MITIGATION_URL} ...")
    log.info(f"Payload: {json.dumps(_build_payload(test_verdict, test_meta), indent=2)}")
    success = send_alert(test_verdict, test_meta)
    if success:
        log.info("Alert delivered ✓")
    else:        log.warning(
            "Alert not delivered — this is expected if the Mitigation Engine "
            "is not running yet. Check MITIGATION_URL in .env"
        )


if __name__ == "__main__":
    _smoke_test()