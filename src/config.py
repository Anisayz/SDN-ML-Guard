"""
config.py — Central configuration for ML Engine
=================================================
Single source of truth for all settings. Every other module imports
from here instead of reading os.getenv() directly.

Override any value by setting the corresponding environment variable
in .env or the shell before starting the application.

Usage:
    from config import cfg
    print(cfg.MITIGATION_URL)
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (silently ignored if not found)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    BASE_DIR      : Path = _ROOT
    DATA_DIR      : Path = _ROOT / "data"
    MODELS_DIR    : Path = _ROOT / "models"
    PROCESSED_DIR : Path = _ROOT / "data" / "processed"
    FEATURE_LIST  : Path = _ROOT / "data" / "feature_list.json"

    # ------------------------------------------------------------------
    # Alert / Mitigation Engine
    # ------------------------------------------------------------------
    MITIGATION_URL  : str   = os.getenv("MITIGATION_URL",  "http://localhost:9000/alert")
    ALERT_TIMEOUT_S : float = float(os.getenv("ALERT_TIMEOUT_S", "2.0"))
    ALERT_RETRIES   : int   = int(os.getenv("ALERT_RETRIES",     "2"))

    # ------------------------------------------------------------------
    # Inference thresholds
    # ------------------------------------------------------------------
    # AE_THRESHOLD: reconstruction error above this → anomaly flagged.
    # Set automatically by evaluate.py optimal F1 sweep.
    # Override in .env after retraining if threshold changes.
    AE_THRESHOLD : float = float(os.getenv("AE_THRESHOLD", "0.003863"))

    # RF_CONF_HIGH: RF confidence >= this → ATTACK regardless of AE.
    # RF_CONF_MIN:  RF confidence below this → low-confidence prediction.
    RF_CONF_HIGH : float = float(os.getenv("RF_CONF_HIGH", "0.70"))
    RF_CONF_MIN  : float = float(os.getenv("RF_CONF_MIN",  "0.50"))

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    BENIGN_LABEL : str = os.getenv("BENIGN_LABEL", "Benign")

    # Verdict → mitigation action mapping
    VERDICT_ACTIONS : dict = None   # set post-init below

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    API_HOST : str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT : int = int(os.getenv("API_PORT", "8000"))

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    # Network interface to listen on (e.g. "eth0", "ens33", "Wi-Fi")
    CAPTURE_INTERFACE   : str   = os.getenv("CAPTURE_INTERFACE",   "eth0")

    # nfstream idle/active timeout in milliseconds
    # A flow is exported after IDLE_TIMEOUT_MS of inactivity
    IDLE_TIMEOUT_MS     : int   = int(os.getenv("IDLE_TIMEOUT_MS",     "120000"))
    ACTIVE_TIMEOUT_MS   : int   = int(os.getenv("ACTIVE_TIMEOUT_MS",   "1800000"))

    # Max flows queued before dropping (back-pressure protection)
    CAPTURE_QUEUE_SIZE  : int   = int(os.getenv("CAPTURE_QUEUE_SIZE",  "1000"))

    # Batch size for predict_batch() calls from capture.py
    # Larger = more efficient, but adds latency before first verdict
    CAPTURE_BATCH_SIZE  : int   = int(os.getenv("CAPTURE_BATCH_SIZE",  "32"))

    # Only send alerts for these verdicts (comma-separated)
    ALERT_VERDICTS      : str   = os.getenv("ALERT_VERDICTS", "ATTACK,SUSPECT,ANOMALY")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL : str = os.getenv("LOG_LEVEL", "INFO")

    def __post_init__(self):
        # Frozen dataclass — use object.__setattr__ to set computed fields
        object.__setattr__(self, "VERDICT_ACTIONS", {
            "ATTACK" : "block",
            "SUSPECT": "alert",
            "ANOMALY": "rate_limit+alert",
            "BENIGN" : "pass",
        })

    def alert_verdicts_set(self) -> set[str]:
        """Return the set of verdicts that should trigger an alert."""
        return {v.strip() for v in self.ALERT_VERDICTS.split(",")}

    def log_summary(self) -> None:
        """Log the active configuration at startup."""
        log.info("=" * 50)
        log.info("ML Engine — Active Configuration")
        log.info("=" * 50)
        log.info(f"  MITIGATION_URL    : {self.MITIGATION_URL}")
        log.info(f"  AE_THRESHOLD      : {self.AE_THRESHOLD}")
        log.info(f"  RF_CONF_HIGH      : {self.RF_CONF_HIGH}")
        log.info(f"  CAPTURE_INTERFACE : {self.CAPTURE_INTERFACE}")
        log.info(f"  CAPTURE_BATCH_SIZE: {self.CAPTURE_BATCH_SIZE}")
        log.info(f"  ALERT_VERDICTS    : {self.ALERT_VERDICTS}")
        log.info(f"  API_HOST:PORT     : {self.API_HOST}:{self.API_PORT}")
        log.info(f"  LOG_LEVEL         : {self.LOG_LEVEL}")
        log.info("=" * 50)


# Module-level singleton — import this everywhere
cfg = Config()