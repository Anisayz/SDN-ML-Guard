"""
heuristics.py — Stateless per-flow rule engine (final savior layer)

Called by predict.py AFTER ML fusion. Each rule inspects a single completed
flow's features and metadata. Rules can only UPGRADE a verdict (BENIGN →
SUSPECT → ANOMALY → ATTACK) — they never downgrade. This ensures the ML
engine's high-confidence decisions are never overridden downward.

Rule design philosophy
──────────────────────
Every rule targets a specific blind spot exposed by real test results:

  • nmap       → low pkt count, many SYN, short duration, sequential ports
  • hping3 UDP → tiny UDP flows with no response
  • hping3 HTTP→ minimal HTTP that looks like benign (caught by timing/ratio)
  • Hydra FTP  → very high SYN count, rapid connections to port 21/22
  • Hydra SSH  → same pattern on port 22
  • Slowloris  → many FIN=0, long-lived HTTP, tiny payload
  • Nikto      → high pkt rate, many URG flags, port 80/443
  • sqlmap/XSS → high byte variance, many PSH flags, port 80/443
  • ping flood → abnormally high ICMP volume

Key naming
──────────
flow dicts arriving here are produced by feature_extractor.flow_to_features()
which outputs CIC-style column names (e.g. "SYN Flag Cnt", "Tot Fwd Pkts").
The _f() helper also accepts the raw nfstream attribute names as aliases so
that callers passing un-converted flow dicts still work correctly.

Environment overrides (all env-var tunable, no redeployment needed):
  HEUR_ENABLED=1                  master switch (default: 1)
  HEUR_SYN_FLOOD_RATIO=0.90       fraction of pkts that are SYN → flood
  HEUR_PORTSCAN_SYNS=5            raw SYN count that suggests a scan
  HEUR_BRUTE_FORCE_PORTS=21,22,23,25,3389,5900
  HEUR_BRUTE_MIN_SYNS=8           min SYN count for brute-force ports
  HEUR_SLOWLORIS_MIN_DUR_S=10     minimum flow duration (seconds)
  HEUR_SLOWLORIS_MAX_BPS=200      maximum bytes/sec for slow-loris
  HEUR_SLOWLORIS_MAX_PKTS=30      maximum total packets
  HEUR_NIKTO_MIN_PPS=20           minimum pkts/sec for scanner
  HEUR_NIKTO_URG_RATIO=0.05       URG flag ratio threshold
  HEUR_SQLI_PSH_RATIO=0.60        PSH flag ratio for injection tools
  HEUR_ICMP_MIN_PKTS=50           ICMP flood minimum packet count
  HEUR_EMPTY_UDP_MAX_BYTES=10     max total bytes for empty UDP probe
  HEUR_HTTP_PROBE_MAX_BPS=50      very slow HTTP that may be a probe
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# ── master switch ────────────────────────────────────────────────────────────
HEUR_ENABLED: bool = os.getenv("HEUR_ENABLED", "1") == "1"

# ── thresholds (all env-tunable) ─────────────────────────────────────────────
_SYN_FLOOD_RATIO:      float     = float(os.getenv("HEUR_SYN_FLOOD_RATIO",    "0.90"))
_PORTSCAN_SYNS:        int       = int(  os.getenv("HEUR_PORTSCAN_SYNS",      "5"))
_BRUTE_FORCE_PORTS:    frozenset = frozenset(
    int(p) for p in os.getenv("HEUR_BRUTE_FORCE_PORTS", "21,22,23,25,3389,5900").split(",") if p.strip()
)
_BRUTE_MIN_SYNS:       int       = int(  os.getenv("HEUR_BRUTE_MIN_SYNS",     "8"))
_SLOWLORIS_MIN_DUR_S:  float     = float(os.getenv("HEUR_SLOWLORIS_MIN_DUR_S","10"))
_SLOWLORIS_MAX_BPS:    float     = float(os.getenv("HEUR_SLOWLORIS_MAX_BPS",  "200"))
_SLOWLORIS_MAX_PKTS:   int       = int(  os.getenv("HEUR_SLOWLORIS_MAX_PKTS", "30"))
_NIKTO_MIN_PPS:        float     = float(os.getenv("HEUR_NIKTO_MIN_PPS",      "20"))
_NIKTO_URG_RATIO:      float     = float(os.getenv("HEUR_NIKTO_URG_RATIO",    "0.05"))
_SQLI_PSH_RATIO:       float     = float(os.getenv("HEUR_SQLI_PSH_RATIO",     "0.60"))
_ICMP_MIN_PKTS:        int       = int(  os.getenv("HEUR_ICMP_MIN_PKTS",      "50"))
_EMPTY_UDP_MAX_BYTES:  int       = int(  os.getenv("HEUR_EMPTY_UDP_MAX_BYTES", "10"))
_HTTP_PROBE_MAX_BPS:   float     = float(os.getenv("HEUR_HTTP_PROBE_MAX_BPS",  "50"))

# ── verdict ordering (higher index = more severe) ────────────────────────────
_SEVERITY: dict[str, int] = {"BENIGN": 0, "ANOMALY": 1, "SUSPECT": 2, "ATTACK": 3}


def _upgrade(current: str, candidate: str) -> str:
    """Return whichever verdict is more severe. Never downgrade."""
    return candidate if _SEVERITY.get(candidate, 0) > _SEVERITY.get(current, 0) else current


# ── feature key aliases ───────────────────────────────────────────────────────
# flow dicts use CIC-style names from feature_extractor.flow_to_features().
# Each entry is (cic_key, *nfstream_fallback_keys) so rules work regardless
# of whether the caller passed a CIC-keyed or raw nfstream-keyed dict.
_ALIASES: dict[str, list[str]] = {
    "Tot Fwd Pkts"    : ["src2dst_packets"],
    "Tot Bwd Pkts"    : ["dst2src_packets"],
    "TotLen Fwd Pkts" : ["src2dst_bytes"],
    "TotLen Bwd Pkts" : ["dst2src_bytes"],
    "SYN Flag Cnt"    : ["bidirectional_syn_packets"],
    "FIN Flag Cnt"    : ["bidirectional_fin_packets"],
    # PSH and URG: prefer bidirectional count; fall back to fwd-only
    "PSH Flag Cnt"    : ["bidirectional_psh_packets", "Fwd PSH Flags", "src2dst_psh_packets"],
    "URG Flag Cnt"    : ["bidirectional_urg_packets", "Fwd URG Flags", "src2dst_urg_packets"],
    "Flow Duration"   : ["bidirectional_duration_ms"],   # see note below
    "Dst Port"        : ["dst_port"],
    "Protocol"        : ["protocol"],
}

# NOTE on Flow Duration:
#   feature_extractor.flow_to_features() converts nfstream ms → µs by doing
#   duration_ms * 1000, so flow["Flow Duration"] is already in microseconds.
#   Rules divide by 1_000_000 to get seconds — this is correct.
#   If "Flow Duration" is missing and the alias "bidirectional_duration_ms"
#   is used instead, the value is in ms and the division gives seconds/1000,
#   producing a duration 1000× too small (10s Slowloris looks like 0.01s).
#   The alias is therefore a last-resort safety net for un-converted dicts;
#   the primary path via flow_to_features() always provides the µs value.


def _f(flow: dict, key: str, default: float = 0.0) -> float:
    """
    Safe feature accessor with alias resolution.

    Lookup order:
      1. CIC key directly (primary — flow_to_features output)
      2. Alias keys in order (fallback for raw nfstream dicts)
      3. default (0.0)

    Always returns a finite float; NaN / ±inf → default.
    """
    def _clean(v: Any) -> float | None:
        try:
            f = float(v)
            return None if (f != f or f == float("inf") or f == float("-inf")) else f
        except (TypeError, ValueError):
            return None

    # 1. Try the key itself
    if key in flow:
        v = _clean(flow[key])
        if v is not None:
            return v

    # 2. Try registered aliases
    for alias in _ALIASES.get(key, []):
        if alias in flow:
            v = _clean(flow[alias])
            if v is not None:
                return v

    return default


# ── individual rules ─────────────────────────────────────────────────────────

def _rule_syn_flood(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    SYN flood: the vast majority of packets carry the SYN flag.
    Catches hping3 and nmap SYN scans.
    """
    total_pkts = _f(flow, "Tot Fwd Pkts") + _f(flow, "Tot Bwd Pkts")
    syn_pkts   = _f(flow, "SYN Flag Cnt")
    if total_pkts < 2:
        return None
    ratio = syn_pkts / total_pkts
    if ratio >= _SYN_FLOOD_RATIO and syn_pkts >= 2:
        return "SUSPECT", f"syn_flood(ratio={ratio:.2f}, syns={int(syn_pkts)})"
    return None


def _rule_portscan(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    Port scan signature: very few packets, several SYNs, nearly zero payload.
    Targets nmap default scan (-sS).
    """
    fwd_pkts    = _f(flow, "Tot Fwd Pkts")
    syn_pkts    = _f(flow, "SYN Flag Cnt")
    total_bytes = _f(flow, "TotLen Fwd Pkts") + _f(flow, "TotLen Bwd Pkts")

    if fwd_pkts <= 4 and syn_pkts >= _PORTSCAN_SYNS and total_bytes < 200:
        return "SUSPECT", f"portscan(fwd_pkts={int(fwd_pkts)}, syns={int(syn_pkts)}, bytes={int(total_bytes)})"
    return None


def _rule_brute_force(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    Brute-force on auth ports (FTP/SSH/RDP/etc): elevated SYN count, small
    payload per packet. Targets Hydra.
    """
    dst_port = int(meta.get("dst_port") or flow.get("Dst Port") or _f(flow, "Dst Port") or 0)
    if dst_port not in _BRUTE_FORCE_PORTS:
        return None

    syn_pkts    = _f(flow, "SYN Flag Cnt")
    total_pkts  = _f(flow, "Tot Fwd Pkts") + _f(flow, "Tot Bwd Pkts")
    total_bytes = _f(flow, "TotLen Fwd Pkts") + _f(flow, "TotLen Bwd Pkts")
    mean_pkt    = (total_bytes / total_pkts) if total_pkts > 0 else 0

    if syn_pkts >= _BRUTE_MIN_SYNS and mean_pkt < 150:
        return "ATTACK", (
            f"brute_force(port={dst_port}, syns={int(syn_pkts)}, "
            f"mean_pkt={mean_pkt:.0f}B)"
        )
    return None


def _rule_slowloris(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    Slowloris HTTP DoS: long-lived connection, near-zero throughput,
    few packets, targeting port 80/443. FIN count low (connection kept open).
    Catches Slowloris and R.U.D.Y.
    """
    dst_port = int(meta.get("dst_port") or flow.get("Dst Port") or _f(flow, "Dst Port") or 0)
    if dst_port not in (80, 443, 8080, 8443):
        return None

    dur_us      = _f(flow, "Flow Duration")      # microseconds (feature_extractor converts ms→µs)
    dur_s       = dur_us / 1_000_000
    total_bytes = _f(flow, "TotLen Fwd Pkts") + _f(flow, "TotLen Bwd Pkts")
    total_pkts  = _f(flow, "Tot Fwd Pkts") + _f(flow, "Tot Bwd Pkts")
    bps         = total_bytes / dur_s if dur_s > 0 else 0
    fin_pkts    = _f(flow, "FIN Flag Cnt")

    if (dur_s >= _SLOWLORIS_MIN_DUR_S
            and bps <= _SLOWLORIS_MAX_BPS
            and total_pkts <= _SLOWLORIS_MAX_PKTS
            and fin_pkts == 0):
        return "ATTACK", (
            f"slowloris(dur={dur_s:.1f}s, bps={bps:.1f}, "
            f"pkts={int(total_pkts)}, fin={int(fin_pkts)})"
        )
    return None


def _rule_web_scanner(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    Web scanner / directory fuzzer (Nikto, dirbuster, gobuster):
    high packet rate, elevated URG flags, targeting HTTP ports.
    """
    dst_port = int(meta.get("dst_port") or flow.get("Dst Port") or _f(flow, "Dst Port") or 0)
    if dst_port not in (80, 443, 8080, 8443):
        return None

    dur_us     = _f(flow, "Flow Duration")
    dur_s      = dur_us / 1_000_000
    if dur_s <= 0:
        return None

    total_pkts = _f(flow, "Tot Fwd Pkts") + _f(flow, "Tot Bwd Pkts")
    urg_pkts   = _f(flow, "URG Flag Cnt")
    pps        = total_pkts / dur_s
    urg_ratio  = urg_pkts / total_pkts if total_pkts > 0 else 0

    if pps >= _NIKTO_MIN_PPS and urg_ratio >= _NIKTO_URG_RATIO:
        return "SUSPECT", (
            f"web_scanner(pps={pps:.1f}, urg_ratio={urg_ratio:.3f})"
        )
    return None


def _rule_injection_tool(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    SQL injection / XSS tool (sqlmap, XSStrike): elevated PSH ratio on HTTP
    ports indicates repeated short request/response bursts with payload.
    Uses bidirectional PSH count (fwd + bwd) for accuracy — fwd-only PSH
    undercounts because server ACKs also carry PSH in pipelined responses.
    """
    dst_port = int(meta.get("dst_port") or flow.get("Dst Port") or _f(flow, "Dst Port") or 0)
    if dst_port not in (80, 443, 8080, 8443):
        return None

    total_pkts = _f(flow, "Tot Fwd Pkts") + _f(flow, "Tot Bwd Pkts")
    if total_pkts < 6:
        return None

    psh_pkts  = _f(flow, "PSH Flag Cnt")   # bidirectional via alias resolution
    psh_ratio = psh_pkts / total_pkts

    if psh_ratio >= _SQLI_PSH_RATIO:
        return "SUSPECT", f"injection_tool(psh_ratio={psh_ratio:.2f}, pkts={int(total_pkts)})"
    return None


def _rule_icmp_flood(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    ICMP flood / ping flood: protocol=1, very high packet count.
    A normal ping is 4–5 packets — anything beyond _ICMP_MIN_PKTS is suspicious.
    """
    protocol = int(meta.get("protocol") or flow.get("Protocol") or _f(flow, "Protocol") or -1)
    if protocol != 1:   # ICMP
        return None

    total_pkts = _f(flow, "Tot Fwd Pkts") + _f(flow, "Tot Bwd Pkts")
    if total_pkts >= _ICMP_MIN_PKTS:
        return "SUSPECT", f"icmp_flood(pkts={int(total_pkts)})"
    return None


def _rule_empty_udp_probe(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    Empty UDP probe (hping3 --udp): tiny total bytes, UDP protocol, no response.
    Benign UDP traffic always has a meaningful payload (DNS, etc.).
    """
    protocol = int(meta.get("protocol") or flow.get("Protocol") or _f(flow, "Protocol") or -1)
    if protocol != 17:   # UDP
        return None

    dst_port = int(meta.get("dst_port") or flow.get("Dst Port") or _f(flow, "Dst Port") or 0)
    # Let DNS/DHCP/mDNS through — they're already whitelisted but be defensive
    if dst_port in (53, 67, 68, 5353):
        return None

    total_bytes = _f(flow, "TotLen Fwd Pkts") + _f(flow, "TotLen Bwd Pkts")
    bwd_pkts    = _f(flow, "Tot Bwd Pkts")   # 0 response packets = no reply

    if total_bytes <= _EMPTY_UDP_MAX_BYTES and bwd_pkts == 0:
        return "SUSPECT", f"empty_udp_probe(bytes={int(total_bytes)}, bwd_pkts=0)"
    return None


def _rule_http_probe(flow: dict, meta: dict) -> tuple[str, str] | None:
    """
    Extremely slow HTTP flow with tiny payload — different from Slowloris
    because duration is shorter but throughput is suspiciously low.
    Catches manual curl probes and some hping3 HTTP patterns.
    """
    dst_port = int(meta.get("dst_port") or flow.get("Dst Port") or _f(flow, "Dst Port") or 0)
    if dst_port not in (80, 443, 8080, 8443):
        return None

    dur_us      = _f(flow, "Flow Duration")
    dur_s       = dur_us / 1_000_000
    if dur_s < 0.5 or dur_s >= _SLOWLORIS_MIN_DUR_S:
        # too short (normal req) or too long (caught by slowloris rule)
        return None

    total_bytes = _f(flow, "TotLen Fwd Pkts") + _f(flow, "TotLen Bwd Pkts")
    bps         = total_bytes / dur_s

    if bps <= _HTTP_PROBE_MAX_BPS and total_bytes < 500:
        return "SUSPECT", f"http_probe(bps={bps:.1f}, bytes={int(total_bytes)}, dur={dur_s:.2f}s)"
    return None


# ── rule registry (order matters: more specific rules first) ─────────────────
_RULES = [
    _rule_brute_force,      # specific port + pattern → ATTACK
    _rule_slowloris,        # specific port + timing → ATTACK
    _rule_syn_flood,        # flag ratio → SUSPECT
    _rule_portscan,         # few pkts + many SYNs → SUSPECT
    _rule_icmp_flood,       # ICMP volume → SUSPECT
    _rule_empty_udp_probe,  # UDP with no payload → SUSPECT
    _rule_web_scanner,      # HTTP port + high pps + URG → SUSPECT
    _rule_injection_tool,   # HTTP port + high PSH ratio → SUSPECT
    _rule_http_probe,       # HTTP port + very low bps → SUSPECT
]


# ── public entry point ───────────────────────────────────────────────────────

def apply_heuristics(
    flow:    dict[str, Any],
    meta:    dict[str, Any],
    verdict: str,
    source:  str,
    flags:   list[str],
) -> tuple[str, str, list[str]]:
    """
    Run all heuristic rules against a single flow.

    Args:
        flow:    CIC-keyed feature dict as produced by feature_extractor.flow_to_features()
        meta:    identity dict (src_ip, dst_ip, src_port, dst_port, protocol)
                 as produced by feature_extractor.flow_to_meta()
        verdict: current ML verdict string
        source:  current ML source string
        flags:   mutable list to accumulate rule names that fired

    Returns:
        (verdict, source, flags) — verdict may have been upgraded.
    """
    if not HEUR_ENABLED:
        return verdict, source, flags

    for rule in _RULES:
        try:
            result = rule(flow, meta)
            if result is None:
                continue
            new_verdict, flag_msg = result
            upgraded = _upgrade(verdict, new_verdict)
            if upgraded != verdict:
                log.debug(
                    "[heuristics] %s: %s → %s  (%s)",
                    rule.__name__, verdict, upgraded, flag_msg,
                )
                verdict = upgraded
                source  = f"HEUR:{rule.__name__}"
            flags.append(f"{rule.__name__}:{flag_msg}")
        except Exception as exc:
            log.warning("[heuristics] Rule %s raised: %s", rule.__name__, exc)

    return verdict, source, flags


def explain(flow: dict[str, Any], meta: dict[str, Any]) -> list[dict]:
    """
    Diagnostic helper: run all rules and return a structured report
    regardless of whether they upgraded the verdict. Useful for debugging
    and for building a detection dashboard.

    Returns a list of dicts: {rule, fired, verdict, detail}
    """
    report = []
    for rule in _RULES:
        try:
            result = rule(flow, meta)
            report.append({
                "rule"   : rule.__name__,
                "fired"  : result is not None,
                "verdict": result[0] if result else None,
                "detail" : result[1] if result else None,
            })
        except Exception as exc:
            report.append({
                "rule"   : rule.__name__,
                "fired"  : False,
                "verdict": None,
                "detail" : f"ERROR: {exc}",
            })
    return report


def debug_flow(flow: dict[str, Any], meta: dict[str, Any]) -> None:
    """
    Print every key value that heuristic rules care about, for both the
    CIC name and its nfstream alias. Call this on a flow that should have
    triggered a rule but didn't.

    Usage:
        from heuristics import debug_flow
        from feature_extractor import flow_to_features, flow_to_meta
        debug_flow(flow_to_features(nf_flow), flow_to_meta(nf_flow))
    """
    keys_of_interest = [
        "Tot Fwd Pkts", "Tot Bwd Pkts",
        "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "SYN Flag Cnt", "FIN Flag Cnt", "PSH Flag Cnt", "URG Flag Cnt",
        "Flow Duration",
        "Dst Port", "Protocol",
    ]
    print("\n── heuristics.debug_flow ──────────────────────────────")
    print(f"  meta: dst_port={meta.get('dst_port')}  protocol={meta.get('protocol')}")
    for k in keys_of_interest:
        v = _f(flow, k)
        aliases = _ALIASES.get(k, [])
        raw = {a: flow.get(a, "<missing>") for a in [k] + aliases}
        print(f"  {k:20s} = {v:>12.2f}   raw={raw}")
    print("── explain ────────────────────────────────────────────")
    for entry in explain(flow, meta):
        status = "FIRED" if entry["fired"] else "skip "
        print(f"  [{status}] {entry['rule']:25s}  {entry['detail'] or ''}")
    print()


if __name__ == "__main__":
    # Quick self-test with synthetic flows
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)-8s %(message)s")

    tests = [
        ("nmap SYN scan",   {"SYN Flag Cnt": 10, "Tot Fwd Pkts": 10, "Tot Bwd Pkts": 1,
                              "TotLen Fwd Pkts": 60, "TotLen Bwd Pkts": 0},
                             {"dst_port": 22, "protocol": 6}),

        ("Hydra FTP",       {"SYN Flag Cnt": 20, "Tot Fwd Pkts": 25, "Tot Bwd Pkts": 5,
                              "TotLen Fwd Pkts": 800, "TotLen Bwd Pkts": 200},
                             {"dst_port": 21, "protocol": 6}),

        ("Slowloris",       {"Flow Duration": 30_000_000, "Tot Fwd Pkts": 8, "Tot Bwd Pkts": 4,
                              "TotLen Fwd Pkts": 300, "TotLen Bwd Pkts": 100, "FIN Flag Cnt": 0},
                             {"dst_port": 80, "protocol": 6}),

        ("hping3 UDP",      {"Tot Fwd Pkts": 5, "Tot Bwd Pkts": 0,
                              "TotLen Fwd Pkts": 5, "TotLen Bwd Pkts": 0},
                             {"dst_port": 12345, "protocol": 17}),

        ("sqlmap",          {"Tot Fwd Pkts": 40, "Tot Bwd Pkts": 20,
                              "PSH Flag Cnt": 38, "TotLen Fwd Pkts": 4000, "TotLen Bwd Pkts": 2000},
                             {"dst_port": 80, "protocol": 6}),

        ("Normal ping",     {"Tot Fwd Pkts": 4, "Tot Bwd Pkts": 4},
                             {"dst_port": 0, "protocol": 1}),

        ("ICMP flood",      {"Tot Fwd Pkts": 500, "Tot Bwd Pkts": 500},
                             {"dst_port": 0, "protocol": 1}),

        ("Normal HTTP",     {"Flow Duration": 200_000, "Tot Fwd Pkts": 10, "Tot Bwd Pkts": 8,
                              "TotLen Fwd Pkts": 5000, "TotLen Bwd Pkts": 20000,
                              "PSH Flag Cnt": 4, "FIN Flag Cnt": 2},
                             {"dst_port": 80, "protocol": 6}),

        # nfstream alias test — same as Hydra FTP but using raw nfstream key names
        ("Hydra FTP (nfstream keys)",
                            {"bidirectional_syn_packets": 20,
                             "src2dst_packets": 25, "dst2src_packets": 5,
                             "src2dst_bytes": 800, "dst2src_bytes": 200},
                             {"dst_port": 21, "protocol": 6}),
    ]

    all_passed = True
    for name, flow, meta in tests:
        verdict, source, flags = apply_heuristics(flow, meta, "BENIGN", "RF+AE", [])
        fired = [f for f in flags]
        print(f"  {name:30s} → {verdict:7s}  src={source:30s}  flags={fired}")

    print("\nAlias resolution test:")
    syn_via_alias = _f({"bidirectional_syn_packets": 42}, "SYN Flag Cnt")
    psh_via_alias = _f({"bidirectional_psh_packets": 7},  "PSH Flag Cnt")
    dur_via_alias = _f({"bidirectional_duration_ms": 500}, "Flow Duration")
    print(f"  SYN via alias = {syn_via_alias}  (expect 42)")
    print(f"  PSH via alias = {psh_via_alias}  (expect 7)")
    print(f"  Duration via alias = {dur_via_alias}  (expect 500 — ms, not µs)")
    assert syn_via_alias == 42.0
    assert psh_via_alias == 7.0
    assert dur_via_alias == 500.0
    print("  All alias assertions passed.")
    