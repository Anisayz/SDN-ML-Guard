from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import cfg
from feature_extractor import flow_to_features, flow_to_meta
from predict import get_engine
from alert import send_alert, _resolve_attack_type

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALERT_DEDUP_WINDOW_S: float = float(
    __import__("os").getenv("ALERT_DEDUP_WINDOW_S", "30")
)


# ---------------------------------------------------------------------------
# Stats counter — thread-safe
# ---------------------------------------------------------------------------
class _Stats:
    def __init__(self):
        self._lock        = threading.Lock()
        self.flows        = 0
        self.attack       = 0
        self.suspect      = 0
        self.anomaly      = 0
        self.benign       = 0
        self.alerts_ok    = 0
        self.alerts_fail  = 0
        self.alerts_dedup = 0
        self.errors       = 0
        self.start_time   = time.time()

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

    def record_dedup(self) -> None:
        with self._lock:
            self.alerts_dedup += 1

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
                f"dedup_suppressed={self.alerts_dedup}  "
                f"errors={self.errors}  fps={fps:.1f}"
            )


stats = _Stats()


# ---------------------------------------------------------------------------
# Alert deduplicator — thread-safe
# ---------------------------------------------------------------------------
class _AlertDeduplicator:
    def __init__(self, window_s: float = 30.0):
        self._lock       = threading.Lock()
        self._window     = window_s
        self._last: dict[tuple, float] = {}
        self._suppressed: dict[tuple, int] = {}

    def should_alert(self, src_ip: str, dst_ip: str, dst_port: int, verdict: str) -> bool:
        if self._window <= 0:
            return True

        key = (src_ip, dst_ip, dst_port, verdict)
        now = time.monotonic()

        with self._lock:
            last = self._last.get(key, 0.0)
            if now - last >= self._window:
                self._last[key] = now
                suppressed = self._suppressed.pop(key, 0)
                if suppressed > 0:
                    log.info(
                        f"[dedup] {verdict} {src_ip} → {dst_ip}:{dst_port}  "
                        f"suppressed {suppressed:,} duplicate alerts in last "
                        f"{self._window:.0f}s window"
                    )
                return True
            else:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return False

    def flush_log(self) -> None:
        with self._lock:
            for (src_ip, dst_ip, dst_port, verdict), count in self._suppressed.items():
                if count > 0:
                    log.info(
                        f"[dedup] {verdict} {src_ip} → {dst_ip}:{dst_port}  "
                        f"suppressed {count:,} duplicate alerts (final flush)"
                    )
            self._suppressed.clear()


_deduplicator = _AlertDeduplicator(window_s=ALERT_DEDUP_WINDOW_S)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class _InferenceWorker(threading.Thread):

    def __init__(self, flow_queue: queue.Queue, dry_run: bool = False):
        super().__init__(daemon=True, name="inference-worker")
        self.flow_queue      = flow_queue
        self.dry_run         = dry_run
        self.engine          = get_engine()
        self._stop_evt       = threading.Event()
        self._alert_verdicts = cfg.alert_verdicts_set()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        log.info("[worker] Inference worker started")
        batch_flows   = []
        batch_metas   = []
        batch_timeout = 0.1

        while not self._stop_evt.is_set():
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

            try:
                verdicts = self.engine.predict_batch(
                    batch_flows,
                    flow_metas=batch_metas,
                )
            except Exception as e:
                log.error(f"[worker] Inference error: {e}")
                stats.record_error()
                batch_flows.clear()
                batch_metas.clear()
                continue

            for verdict, meta in zip(verdicts, batch_metas):
                stats.record(verdict.verdict)

                # Resolve human-readable attack type for logging and alerting
                attack_type = _resolve_attack_type(verdict.to_dict(), meta)

                log.info(
                    f"[capture] {verdict.verdict:7s}  "
                    f"{attack_type:30s}  "          # ← attack type, not raw RF label
                    f"conf={verdict.confidence:.3f}  "
                    f"ae={verdict.anomaly_score:.4f}  "
                    f"src={meta.get('src_ip')}:{meta.get('src_port')} → "
                    f"dst={meta.get('dst_ip')}:{meta.get('dst_port')}  "
                    f"syn={meta.get('src_syn_packets', '?')}  "
                    f"bwd_bytes={meta.get('dst2src_bytes', '?')}"
                )

                if verdict.verdict not in self._alert_verdicts:
                    continue

                src_ip   = meta.get("src_ip",  "")
                dst_ip   = meta.get("dst_ip",  "")
                dst_port = meta.get("dst_port", 0)

                if not _deduplicator.should_alert(src_ip, dst_ip, dst_port, verdict.verdict):
                    stats.record_dedup()
                    continue

                log.info(
                    f"[alert→] {verdict.verdict}  "
                    f"{src_ip} → {dst_ip}:{dst_port}  "
                    f"label={attack_type}  conf={verdict.confidence:.3f}"
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
    try:
        from nfstream import NFStreamer
    except ImportError:
        log.error("nfstream is not installed. Install it with: pip install nfstream")
        sys.exit(1)

    log.info("=" * 50)
    log.info("ML Engine — Live Capture")
    log.info("=" * 50)
    log.info(f"  Interface   : {interface}")
    log.info(f"  Batch size  : {cfg.CAPTURE_BATCH_SIZE}")
    log.info(f"  Dry run     : {dry_run}")
    log.info(f"  Alerting    : {cfg.MITIGATION_URL}")
    log.info(f"  Verdicts    : {cfg.ALERT_VERDICTS}")
    log.info(f"  Dedup window: {ALERT_DEDUP_WINDOW_S}s  (0 = disabled)")
    log.info("=" * 50)

    if dry_run:
        log.warning("[capture] DRY RUN — alerts will NOT be sent")

    flow_queue = queue.Queue(maxsize=cfg.CAPTURE_QUEUE_SIZE)
    worker     = _InferenceWorker(flow_queue, dry_run=dry_run)
    reporter   = _StatsReporter(interval_s=60)
    worker.start()
    reporter.start()

    _shutdown = threading.Event()
    def _handle_signal(sig, frame):
        log.info(f"[capture] Signal {sig} received — shutting down ...")
        _shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info(f"[capture] Starting nfstream on '{interface}' ...")
    try:
        streamer = NFStreamer(
            source               = interface,
            idle_timeout         = cfg.IDLE_TIMEOUT_MS,
            active_timeout       = cfg.ACTIVE_TIMEOUT_MS,
            statistical_analysis = True,
            splt_analysis        = 0,
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
        _deduplicator.flush_log()
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