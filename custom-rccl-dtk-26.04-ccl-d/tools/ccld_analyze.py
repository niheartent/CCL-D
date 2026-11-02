#!/usr/bin/env python3
"""Group CCL-D records and detect collective mismatches and rank gaps."""
import json
import pathlib
import time
from collections import defaultdict

FIELDS = ("op", "count", "bytes", "datatype", "root")

def load_records(paths):
    records = []
    for path in paths:
        try:
            with pathlib.Path(path).open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a partially written final line
                    if record.get("phase") in ("enqueue", "complete"):
                        records.append(record)
        except FileNotFoundError:
            continue
    return records

def analyze(records, *, now_ns=None, timeout_ms=2000, slow_skew_ms=2):
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    groups = defaultdict(list)
    for record in records:
        if record.get("phase", "enqueue") != "enqueue":
            continue
        key = (str(record.get("comm_id", "unknown")),
               str(record.get("counter", record.get("trace_id", "unknown"))))
        groups[key].append(record)

    results = []
    for (comm_id, counter), rows in sorted(groups.items()):
        ranks = sorted({int(row.get("rank", -1)) for row in rows})
        expected = max((int(row.get("nranks", 1)) for row in rows), default=1)
        nnodes = max((int(row.get("nnodes", 1)) for row in rows), default=1)
        nnodes = max((int(row.get("nnodes", 1)) for row in rows), default=1)
        expected_ranks = list(range(expected))
        missing = sorted(set(expected_ranks) - set(ranks))
        mismatches = {field: sorted({str(row.get(field)) for row in rows})
                      for field in FIELDS if len({str(row.get(field)) for row in rows}) > 1}
        times = [int(row["timestamp_ns"]) for row in rows if row.get("timestamp_ns") is not None]
        age_ms = max(0, (now_ns - min(times)) / 1_000_000) if times else 0
        skew_ms = (max(times) - min(times)) / 1_000_000 if len(times) > 1 else 0

        if mismatches:
            status = "MISMATCH"
        elif missing and age_ms >= timeout_ms:
            status = "MISSING_RANK"
        elif missing:
            status = "PENDING"
        elif nnodes <= 1 and skew_ms >= slow_skew_ms:
            # This is only host enqueue timing.  It is not a send/recv-rate
            # measurement and must not be presented as a communication slow diagnosis.
            status = "ENQUEUE_SKEW"
        else:
            status = "OK"
        results.append({"comm_id": comm_id, "counter": counter,
                        "trace_id": rows[0].get("trace_id"), "status": status,
                        "ranks": ranks, "expected_ranks": expected_ranks,
                        "missing_ranks": missing, "mismatches": mismatches,
                        "nnodes": nnodes, "skew_ms": round(skew_ms, 3),
                        "age_ms": round(age_ms, 1),
                        "records": len(rows)})
    return results

def analyze_completions(records):
    """Aggregate device completion records by communicator, operation and rank."""
    groups = defaultdict(list)
    for record in records:
        if record.get("phase") != "complete":
            continue
        key = (str(record.get("comm_id", "unknown")),
               str(record.get("counter", "unknown")))
        groups[key].append(record)
    results = []
    for (comm_id, counter), rows in sorted(groups.items()):
        by_rank = defaultdict(lambda: {"send_count": 0, "recv_count": 0, "channels": 0})
        for row in rows:
            rank = int(row.get("rank", -1))
            by_rank[rank]["send_count"] += int(row.get("send_count", 0))
            by_rank[rank]["recv_count"] += int(row.get("recv_count", 0))
            by_rank[rank]["channels"] += 1
        expected = max((int(row.get("nranks", 1)) for row in rows), default=1)
        nnodes = max((int(row.get("nnodes", 1)) for row in rows), default=1)
        ranks = sorted(by_rank)
        missing = sorted(set(range(expected)) - set(ranks))
        send_values = sorted({v["send_count"] for v in by_rank.values()})
        recv_values = sorted({v["recv_count"] for v in by_rank.values()})
        algorithms = sorted({int(row.get("algorithm", -1)) for row in rows})
        protocols = sorted({int(row.get("protocol", -1)) for row in rows})
        # Ring ranks have symmetric degree. Tree and CollNet ranks may have
        # different degrees; those require topology-level grouping before a
        # count mismatch can be declared.
        count_mismatch = len(send_values) > 1 or len(recv_values) > 1
        # Device counters are currently channel aggregates.  Even ring
        # collectives can legitimately produce different per-rank totals
        # (the number of local send/recv edges depends on the selected
        # channel/topology).  Until edge/peer metadata is emitted, do not
        # turn such a difference into a false fault.  Synthetic records that
        # omit algorithm metadata ([-1]) retain the strict mismatch check.
        topology_pending = count_mismatch and (
            nnodes > 1 or any(a not in (-1, 0) for a in algorithms) or 0 in algorithms)
        status = "MISSING_COMPLETION" if missing else (
            "COUNT_MISMATCH" if count_mismatch and not topology_pending else
            "TOPOLOGY_PENDING" if topology_pending else "OK")
        results.append({"comm_id": comm_id, "counter": counter, "status": status,
                        "ranks": ranks, "expected_ranks": list(range(expected)),
                        "missing_ranks": missing, "send_counts": send_values,
                        "recv_counts": recv_values, "per_rank": dict(by_rank),
                        "algorithms": algorithms, "protocols": protocols,
                        "nnodes": nnodes,
                        "records": len(rows)})
    return results
