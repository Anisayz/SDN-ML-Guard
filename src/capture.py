"""
capture.py — Live network capture and inference pipeline
=========================================================
Listens on a network interface using nfstream, classifies completed
flows with the ML engine, and dispatches non-BENIGN verdicts to alert.py.

Usage:
    python src/capture.py                        # uses CAPTURE_INTERFACE from .env
    python src/capture.py --interface eth0       # override interface
    python src/capture.py --interface eth0 --dry-run  # no alerts sent

What it does:
    1. nfstream captures packets and assembles them into flows
    2. Each completed flow is converted to features (feature_extractor.py)
    3. Features are batched and sent to predict.py (RF + AE)
    4. Non-BENIGN verdicts are sent to alert.py → Mitigation Engine
    5. Statistics are logged every 60 seconds

Flow of a single packet:
    NIC → nfstream → flow_to_features() → predict_batch() → _fuse() → send_alert()
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import cfg
from feature_extractor import flow_to_features, flow_to_meta
from predict import get_engine
from alert import send_alert

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stats counter — thread-safe
# ---------------------------------------------------------------------------
class _Stats:
    def __init__(self):
        self._lock     = threading.Lock()
        self.flows     = 0
        self.attack    = 0
        self.suspect   = 0
        self.anomaly   = 0
        self.benign    = 0
        self.alerts_ok = 0
        self.alerts_fail = 0
        self.errors    = 0
        self.start_time = time.time()

    def record(self, verdict: str) -> None:
        with self._lock:
            self.flows += 1
            if   verdict == "ATTACK":  self.attack  += 1
            elif verdict == "SUSPECT": self.suspect += 1
            elif verdict == "ANOMALY": self.anomaly += 1
            else:                      self.benign  += 1

    def record_alert(self, ok: bool) -> None:
        with self._lock:
            if ok: self.alerts_ok   += 1
            else:  self.alerts_fail += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def log(self) -> None:
        with self._lock:
            elapsed = time.time() - self.start_time
            fps     = self.flows / elapsed if elapsed > 0 else 0
            log.info(
                f"[stats] flows={self.flows:,}  "
                f"ATTACK={self.attack}  SUSPECT={self.suspect}  "
                f"ANOMALY={self.anomaly}  BENIGN={self.benign}  "
                f"alerts_ok={self.alerts_ok}  alerts_fail={self.alerts_fail}  "
                f"errors={self.errors}  fps={fps:.1f}"
            )


stats = _Stats()


# ---------------------------------------------------------------------------
# Worker thread — processes flows from the queue
# ---------------------------------------------------------------------------
class _InferenceWorker(threading.Thread):
    """
    Pulls completed flows from the queue in batches,
    runs predict_batch(), dispatches alerts.
    """

    def __init__(
        self,
        flow_queue: queue.Queue,
        dry_run: bool = False,
    ):
        super().__init__(daemon=True, name="inference-worker")
        self.flow_queue = flow_queue
        self.dry_run    = dry_run
        self.engine     = get_engine()
        self._stop_evt  = threading.Event()
        self._alert_verdicts = cfg.alert_verdicts_set()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        log.info("[worker] Inference worker started")
        batch_flows  = []
        batch_metas  = []
        batch_timeout = 0.1   # seconds — flush partial batch after this

        while not self._stop_evt.is_set():
            # Drain queue into batch
            deadline = time.monotonic() + batch_timeout
            while time.monotonic() < deadline:
                try:
                    flow_obj = self.flow_queue.get(timeout=0.02)
                    features = flow_to_features(flow_obj)
                    meta     = flow_to_meta(flow_obj)
                    batch_flows.append(features)
                    batch_metas.append(meta)
                    self.flow_queue.task_done()

                    if len(batch_flows) >= cfg.CAPTURE_BATCH_SIZE:
                        break
                except queue.Empty:
                    break

            if not batch_flows:
                continue

            # Run inference on batch
            try:
                verdicts = self.engine.predict_batch(batch_flows)
            except Exception as e:
                log.error(f"[worker] Inference error: {e}")
                stats.record_error()
                batch_flows.clear()
                batch_metas.clear()
                continue

            # Process results
            for verdict, meta in zip(verdicts, batch_metas):
                stats.record(verdict.verdict)

                if verdict.verdict in self._alert_verdicts:
                    log.info(
                        f"[capture] {verdict.verdict:7s}  "
                        f"{verdict.label:30s}  "
                        f"conf={verdict.confidence:.3f}  "
                        f"ae={verdict.anomaly_score:.4f}  "
                        f"src={meta.get('src_ip')}:{meta.get('src_port')} → "
                        f"dst={meta.get('dst_ip')}:{meta.get('dst_port')}"
                    )
                    if not self.dry_run:
                        ok = send_alert(verdict.to_dict(), meta)
                        stats.record_alert(ok)

            batch_flows.clear()
            batch_metas.clear()

        log.info("[worker] Inference worker stopped")


# ---------------------------------------------------------------------------
# Stats reporter thread
# ---------------------------------------------------------------------------
class _StatsReporter(threading.Thread):
    def __init__(self, interval_s: int = 60):
        super().__init__(daemon=True, name="stats-reporter")
        self.interval_s = interval_s
        self._stop_evt  = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.wait(self.interval_s):
            stats.log()


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------
def run_capture(interface: str, dry_run: bool = False) -> None:
    """
    Start nfstream on the given interface and process flows until interrupted.

    Args:
        interface: network interface name (e.g. "eth0", "ens33", "Wi-Fi")
        dry_run:   if True, run inference but do not send alerts
    """
    try:
        from nfstream import NFStreamer
    except ImportError:
        log.error(
            "nfstream is not installed. Install it with: pip install nfstream"
        )
        sys.exit(1)

    log.info("=" * 50)
    log.info("ML Engine — Live Capture")
    log.info("=" * 50)
    log.info(f"  Interface  : {interface}")
    log.info(f"  Batch size : {cfg.CAPTURE_BATCH_SIZE}")
    log.info(f"  Dry run    : {dry_run}")
    log.info(f"  Alerting   : {cfg.MITIGATION_URL}")
    log.info(f"  Verdicts   : {cfg.ALERT_VERDICTS}")
    log.info("=" * 50)

    if dry_run:
        log.warning("[capture] DRY RUN — alerts will NOT be sent")

    # Flow queue — nfstream pushes flows in, worker pulls them out
    flow_queue = queue.Queue(maxsize=cfg.CAPTURE_QUEUE_SIZE)

    # Start worker and stats reporter
    worker   = _InferenceWorker(flow_queue, dry_run=dry_run)
    reporter = _StatsReporter(interval_s=60)
    worker.start()
    reporter.start()

    # Graceful shutdown on SIGINT / SIGTERM
    _shutdown = threading.Event()
    def _handle_signal(sig, frame):
        log.info(f"[capture] Signal {sig} received — shutting down ...")
        _shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Start nfstream
    log.info(f"[capture] Starting nfstream on '{interface}' ...")
    try:
        streamer = NFStreamer(
            source               = interface,
            idle_timeout         = cfg.IDLE_TIMEOUT_MS,
            active_timeout       = cfg.ACTIVE_TIMEOUT_MS,
            statistical_analysis = True,    # enables CICFlowMeter stats
            splt_analysis        = 0,       # disable SPLT (not needed)
            n_dissections        = 20,
        )
    except Exception as e:
        log.error(f"[capture] Failed to start nfstream on '{interface}': {e}")
        log.error("Check that the interface exists and you have capture permissions.")
        sys.exit(1)

    log.info("[capture] Listening for flows ...")
    try:
        for flow in streamer:
            if _shutdown.is_set():
                break
            try:
                flow_queue.put(flow, timeout=1.0)
            except queue.Full:
                log.warning("[capture] Queue full — dropping flow")
                stats.record_error()

    except Exception as e:
        log.error(f"[capture] nfstream error: {e}")
    finally:
        log.info("[capture] Stopping workers ...")
        worker.stop()
        reporter.stop()
        worker.join(timeout=5)
        stats.log()
        log.info("[capture] Shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Engine — Live network capture and inference"
    )
    parser.add_argument(
        "--interface", "-i",
        type=str,
        default=cfg.CAPTURE_INTERFACE,
        help=f"Network interface to capture on (default: {cfg.CAPTURE_INTERFACE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference but do not send alerts to the Mitigation Engine",
    )
    args = parser.parse_args()

    cfg.log_summary()
    run_capture(interface=args.interface, dry_run=args.dry_run)


if __name__ == "__main__":
    main()