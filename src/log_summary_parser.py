#!/usr/bin/env python3
"""
DeepSpeed log_summary parser
Usage:
    python log_summary_parser.py <log_file>
    deepspeed_trainer | python log_summary_parser.py     (reads from stdin)
"""

import sys
import re
from collections import defaultdict

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_size(s: str) -> float:
    """Convert human-readable sizes like '512.0 B', '1.5 KB', '24.54 MB' to bytes."""
    s = s.strip()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    m = re.match(r"([\d.]+)\s*(B|KB|MB|GB)", s)
    if not m:
        return 0.0
    return float(m.group(1)) * units[m.group(2)]

def fmt_size(b: float) -> str:
    """Format bytes back to the most readable unit."""
    for unit, threshold in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]:
        if b >= threshold:
            return f"{b / threshold:.2f} {unit}"
    return f"{b:.2f} B"

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

# ── parser ────────────────────────────────────────────────────────────────────

HEADER_RE = re.compile(
    r"Comm\. Op\s+Message Size\s+Count\s+Total Latency"
)

# Matches data rows: optional op name, then 5–6 numeric columns
ROW_RE = re.compile(
    r"^\s*"
    r"(?P<op>[A-Za-z_][A-Za-z0-9_ ]*?)?\s+"   # optional op name
    r"(?P<msg_size>[\d.]+ (?:B|KB|MB|GB))\s+"
    r"(?P<count>\d+)\s+"
    r"(?P<total_lat>[\d.]+)\s+"
    r"(?P<avg_lat>[\d.]+)\s+"
    r"(?P<tput_avg>[\d.]+)\s+"
    r"(?P<busbw_avg>[\d.]+)"
)

# Simpler row without op name (continuation lines)
CONT_ROW_RE = re.compile(
    r"^\s+"
    r"(?P<msg_size>[\d.]+ (?:B|KB|MB|GB))\s+"
    r"(?P<count>\d+)\s+"
    r"(?P<total_lat>[\d.]+)\s+"
    r"(?P<avg_lat>[\d.]+)\s+"
    r"(?P<tput_avg>[\d.]+)\s+"
    r"(?P<busbw_avg>[\d.]+)"
)

OP_NAME_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)$")  # standalone op name line


def parse(lines: list[str]) -> dict:
    """Return {op_name: [row_dict, ...]}."""
    ops: dict[str, list[dict]] = defaultdict(list)
    current_op = None
    in_summary = False

    for line in lines:
        # detect start of summary block
        if HEADER_RE.search(line):
            in_summary = True
            continue
        if not in_summary:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        # bare op-name line  (e.g. "broadcast")
        if OP_NAME_RE.match(stripped):
            current_op = stripped
            continue

        # continuation data row (leading spaces, no op name)
        m = CONT_ROW_RE.match(line)
        if m and current_op:
            ops[current_op].append({
                "msg_size_raw": m.group("msg_size"),
                "msg_size_b":   parse_size(m.group("msg_size")),
                "count":        int(m.group("count")),
                "total_lat":    float(m.group("total_lat")),
                "avg_lat":      float(m.group("avg_lat")),
                "tput_avg":     float(m.group("tput_avg")),
                "busbw_avg":    float(m.group("busbw_avg")),
            })
            continue

        # row that starts with op name on same line
        m = ROW_RE.match(line)
        if m:
            op = m.group("op").strip() if m.group("op") else current_op
            if op:
                current_op = op
            if current_op:
                ops[current_op].append({
                    "msg_size_raw": m.group("msg_size"),
                    "msg_size_b":   parse_size(m.group("msg_size")),
                    "count":        int(m.group("count")),
                    "total_lat":    float(m.group("total_lat")),
                    "avg_lat":      float(m.group("avg_lat")),
                    "tput_avg":     float(m.group("tput_avg")),
                    "busbw_avg":    float(m.group("busbw_avg")),
                })

    return dict(ops)

# ── reporting ─────────────────────────────────────────────────────────────────

SEP  = "─" * 90
SEP2 = "═" * 90

def print_op_summary(op: str, rows: list[dict]):
    counts      = [r["count"]     for r in rows]
    total_lats  = [r["total_lat"] for r in rows]
    avg_lats    = [r["avg_lat"]   for r in rows]
    tputs       = [r["tput_avg"]  for r in rows]
    busbws      = [r["busbw_avg"] for r in rows]
    msg_bytes   = [r["msg_size_b"] * r["count"] for r in rows]  # weighted volume

    total_count    = sum(counts)
    total_lat_sum  = sum(total_lats)
    total_vol      = sum(msg_bytes)

    # weighted averages (weighted by count)
    w_avg_lat  = sum(r["avg_lat"]  * r["count"] for r in rows) / total_count if total_count else 0
    w_tput     = sum(r["tput_avg"] * r["count"] for r in rows) / total_count if total_count else 0
    w_busbw    = sum(r["busbw_avg"]* r["count"] for r in rows) / total_count if total_count else 0

    print(f"\n  {'Operation:':<22} {op}")
    print(f"  {'Unique msg sizes:':<22} {len(rows)}")
    print(SEP)

    # per-size breakdown
    hdr = f"  {'Msg Size':<14} {'Count':>8}  {'Total Lat (ms)':>16}  {'Avg Lat (ms)':>14}  {'tput_avg (Gbps)':>16}  {'busbw_avg (Gbps)':>17}"
    print(hdr)
    print("  " + "·" * 86)
    for r in rows:
        print(f"  {r['msg_size_raw']:<14} {r['count']:>8}  {r['total_lat']:>16.2f}  {r['avg_lat']:>14.2f}  {r['tput_avg']:>16.4f}  {r['busbw_avg']:>17.4f}")

    print("  " + "·" * 86)
    # totals row
    print(f"  {'TOTAL':<14} {total_count:>8}  {total_lat_sum:>16.2f}  {'':>14}  {'':>16}  {'':>17}")

    print(SEP)
    # statistics block
    print(f"  {'Statistic':<28} {'Avg Lat (ms)':>14}  {'tput (Gbps)':>13}  {'busbw (Gbps)':>14}")
    print("  " + "·" * 66)
    print(f"  {'Weighted mean (by count)':<28} {w_avg_lat:>14.3f}  {w_tput:>13.4f}  {w_busbw:>14.4f}")
    print(f"  {'Std dev (across sizes)':<28} {std(avg_lats):>14.3f}  {std(tputs):>13.4f}  {std(busbws):>14.4f}")
    print(f"  {'Min':<28} {min(avg_lats):>14.3f}  {min(tputs):>13.4f}  {min(busbws):>14.4f}")
    print(f"  {'Max':<28} {max(avg_lats):>14.3f}  {max(tputs):>13.4f}  {max(busbws):>14.4f}")
    print(f"\n  Total data volume moved: {fmt_size(total_vol)}")


def print_global_summary(ops: dict):
    print(f"\n{SEP2}")
    print("  GLOBAL SUMMARY ACROSS ALL OPERATIONS")
    print(SEP2)

    hdr = f"  {'Op':<22} {'Total Count':>12}  {'Total Lat (ms)':>16}  {'W.Avg Lat (ms)':>15}  {'W.Avg tput':>11}  {'W.Avg busbw':>12}  {'Data Volume':>12}"
    print(hdr)
    print("  " + "─" * 88)

    grand_count = 0
    grand_lat   = 0.0
    grand_vol   = 0.0

    for op, rows in ops.items():
        total_count  = sum(r["count"] for r in rows)
        total_lat    = sum(r["total_lat"] for r in rows)
        vol          = sum(r["msg_size_b"] * r["count"] for r in rows)
        w_avg_lat    = sum(r["avg_lat"]   * r["count"] for r in rows) / total_count if total_count else 0
        w_tput       = sum(r["tput_avg"]  * r["count"] for r in rows) / total_count if total_count else 0
        w_busbw      = sum(r["busbw_avg"] * r["count"] for r in rows) / total_count if total_count else 0
        skipped      = total_count == 0 or total_lat == 0

        flag = "  (zeros – skipped in totals)" if skipped else ""
        print(f"  {op:<22} {total_count:>12}  {total_lat:>16.2f}  {w_avg_lat:>15.3f}  {w_tput:>11.4f}  {w_busbw:>12.4f}  {fmt_size(vol):>12}{flag}")

        if not skipped:
            grand_count += total_count
            grand_lat   += total_lat
            grand_vol   += vol

    print("  " + "─" * 88)
    print(f"  {'GRAND TOTAL':<22} {grand_count:>12}  {grand_lat:>16.2f}  {'':>15}  {'':>11}  {'':>12}  {fmt_size(grand_vol):>12}")
    print(SEP2)


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    ops = parse(lines)

    if not ops:
        print("No DeepSpeed log_summary block found in input.")
        sys.exit(1)

    print(SEP2)
    print("  DEEPSPEED COMMUNICATION PROFILER – PARSED SUMMARY")
    print(SEP2)

    for op, rows in ops.items():
        print_op_summary(op, rows)

    print_global_summary(ops)


if __name__ == "__main__":
    main()