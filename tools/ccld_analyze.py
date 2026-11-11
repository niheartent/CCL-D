#!/usr/bin/env python3
"""Group CCL-D records and detect collective mismatches and rank gaps."""
import json
import pathlib
import time
from collections import defaultdict
from statistics import median

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

def analyze_completions(records, *, slow_rate_ratio=0.5):
    """Aggregate completion records and detect missing completion/count/rate outliers.

    ``start_timestamp`` and ``end_timestamp`` are device clock ticks emitted by
    RCCL.  We only compare rates inside one collective, so no cross-node clock
    synchronization is required.  A slow-rate result is advisory and is only
    emitted when every participating rank has a usable duration.
    """
    groups = defaultdict(list)
    for record in records:
        if record.get("phase") != "complete":
            continue
        key = (str(record.get("comm_id", "unknown")),
               str(record.get("counter", "unknown")))
        groups[key].append(record)
    results = []
    for (comm_id, counter), rows in sorted(groups.items()):
        by_rank = defaultdict(lambda: {"send_count": 0, "recv_count": 0,
                                       "channels": 0, "complete": True,
                                       "start_timestamp": None,
                                       "end_timestamp": None})
        for row in rows:
            rank = int(row.get("rank", -1))
            item = by_rank[rank]
            item["send_count"] += int(row.get("send_count", 0))
            item["recv_count"] += int(row.get("recv_count", 0))
            item["channels"] += 1
            item["complete"] = item["complete"] and row.get("complete", True) is not False
            start = row.get("start_timestamp")
            end = row.get("end_timestamp")
            if start is not None:
                start = int(start)
                item["start_timestamp"] = start if item["start_timestamp"] is None else min(item["start_timestamp"], start)
            if end is not None:
                end = int(end)
                item["end_timestamp"] = end if item["end_timestamp"] is None else max(item["end_timestamp"], end)
        expected = max((int(row.get("nranks", 1)) for row in rows), default=1)
        nnodes = max((int(row.get("nnodes", 1)) for row in rows), default=1)
        ranks = sorted(by_rank)
        missing = sorted(set(range(expected)) - set(ranks))
        send_values = sorted({v["send_count"] for v in by_rank.values()})
        recv_values = sorted({v["recv_count"] for v in by_rank.values()})
        algorithms = sorted({int(row.get("algorithm", -1)) for row in rows})
        protocols = sorted({int(row.get("protocol", -1)) for row in rows})
        op_counts = sorted({str(row.get("op_count")) for row in rows
                            if row.get("op_count") is not None})
        topology_edges = sorted({
            (int(row["rank"]), int(row["send_rank"]), int(row["recv_rank"]),
             int(row.get("channel_base", -1)))
            for row in rows if row.get("send_rank") is not None and row.get("recv_rank") is not None
        })
        edge_values = defaultdict(lambda: {"send_count": 0, "recv_count": 0,
                                           "records": 0, "start_timestamp": None,
                                           "end_timestamp": None})
        for row in rows:
            if row.get("send_rank") is None or row.get("recv_rank") is None:
                continue
            edge = (int(row["rank"]), int(row["send_rank"]),
                    int(row["recv_rank"]), int(row.get("channel_base", -1)))
            item = edge_values[edge]
            item["send_count"] += int(row.get("send_count", 0))
            item["recv_count"] += int(row.get("recv_count", 0))
            item["records"] += 1
            for field, reducer in (("start_timestamp", min), ("end_timestamp", max)):
                value = row.get(field)
                if value is not None:
                    value = int(value)
                    item[field] = value if item[field] is None else reducer(item[field], value)
        topology_groups = []
        for edge in sorted(edge_values):
            item = edge_values[edge]
            duration = (item["end_timestamp"] - item["start_timestamp"]
                        if item["start_timestamp"] is not None and
                        item["end_timestamp"] is not None and
                        item["end_timestamp"] > item["start_timestamp"] else None)
            topology_groups.append({"rank": edge[0], "send_rank": edge[1],
                                   "recv_rank": edge[2], "channel_base": edge[3],
                                   "send_count": item["send_count"],
                                   "recv_count": item["recv_count"],
                                   "duration": duration,
                                   "send_rate": (item["send_count"] / duration
                                                  if duration else None),
                                   "recv_rate": (item["recv_count"] / duration
                                                  if duration else None),
                                   "records": item["records"]})
        topology_analysis = {"status": "NO_TOPOLOGY"}
        # RCCL enum values: Tree=0, Ring=1.  Collective completion counters
        # are channel aggregates, not per-edge counts; use actual topology
        # links for structure and per-rank/channel rates for performance.
        ring_rows = [r for r in rows if int(r.get("algorithm", -1)) == 1]
        tree_rows = [r for r in rows if int(r.get("algorithm", -1)) == 0]
        if ring_rows:
            if any(r.get("ring_prev") is None or r.get("ring_next") is None for r in ring_rows):
                topology_analysis = {"status": "RING_TOPOLOGY_PENDING",
                                     "reason": "completion records lack ring_prev/ring_next"}
            else:
                by_channel = defaultdict(list)
                for row in ring_rows:
                    by_channel[int(row.get("channel", -1))].append(row)
                asymmetric = []
                for channel, channel_rows in by_channel.items():
                    by_rank_row = {int(r["rank"]): r for r in channel_rows}
                    ranks = set(by_rank_row)
                    bad = (len(by_rank_row) != len(channel_rows) or
                           any(int(r["ring_prev"]) not in ranks or int(r["ring_next"]) not in ranks
                               for r in channel_rows))
                    if not bad:
                        bad = any(int(by_rank_row[int(r["ring_next"])]["ring_prev"]) != int(r["rank"])
                                  or int(by_rank_row[int(r["ring_prev"])]["ring_next"]) != int(r["rank"])
                                  for r in channel_rows)
                    rates = []
                    for r in channel_rows:
                        start, end = r.get("start_timestamp"), r.get("end_timestamp")
                        duration = int(end) - int(start) if start is not None and end is not None else 0
                        if duration > 0 and int(r.get("send_count", 0)) > 0:
                            rates.append(int(r["send_count"]) / duration)
                    if len(rates) >= 2 and min(rates) > 0 and max(rates) / min(rates) >= 2:
                        bad = True
                    if bad:
                        asymmetric.append(channel)
                topology_analysis = {"status": "RING_ASYMMETRY" if asymmetric else "RING_SYMMETRIC",
                                     "channels": sorted(by_channel),
                                     "asymmetric_channels": sorted(asymmetric)}
        elif tree_rows:
            required = ("tree_up", "tree_down0", "tree_down1", "tree_down2", "tree_depth")
            if any(any(r.get(field) is None for field in required) for r in tree_rows):
                topology_analysis = {"status": "TREE_LEVEL_PENDING",
                                     "reason": "completion records lack tree peers or depth"}
            else:
                by_level = defaultdict(list)
                for row in tree_rows:
                    start, end = row.get("start_timestamp"), row.get("end_timestamp")
                    duration = int(end) - int(start) if start is not None and end is not None else 0
                    if duration > 0 and int(row.get("send_count", 0)) > 0:
                        key = (int(row.get("channel", -1)), int(row["tree_depth"]))
                        by_level[key].append(int(row["send_count"]) / duration)
                asymmetric = []
                for (channel, depth), rates in by_level.items():
                    if len(rates) >= 2 and min(rates) > 0 and max(rates) / min(rates) >= 2:
                        asymmetric.append({"channel": channel, "depth": depth})
                topology_analysis = {"status": "TREE_ASYMMETRY" if asymmetric else "TREE_SYMMETRIC",
                                     "channel_depths": [
                                         {"channel": c, "depth": d, "ranks": len(rates)}
                                         for (c, d), rates in sorted(by_level.items())],
                                     "asymmetric_levels": asymmetric}
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
            nnodes > 1 or any(a != -1 for a in algorithms))
        incomplete = sorted(rank for rank, value in by_rank.items() if not value["complete"])
        for value in by_rank.values():
            start = value["start_timestamp"]
            end = value["end_timestamp"]
            duration = end - start if start is not None and end is not None and end > start else None
            value["duration"] = duration
            value["send_rate"] = (value["send_count"] / duration) if duration else None
            value["recv_rate"] = (value["recv_count"] / duration) if duration else None
        usable_rates = [v["send_rate"] for v in by_rank.values() if v["send_rate"] is not None and v["send_rate"] > 0]
        slow_ranks = []
        if len(usable_rates) == len(by_rank) and len(usable_rates) >= 2:
            baseline = sorted(usable_rates)[len(usable_rates) // 2]
            slow_ranks = sorted(rank for rank, value in by_rank.items()
                                if value["send_rate"] < baseline * slow_rate_ratio)
        topology_fault = topology_analysis["status"] in ("RING_ASYMMETRY", "TREE_ASYMMETRY")
        status = "MISSING_COMPLETION" if missing or incomplete else (
            "COMM_SLOW" if slow_ranks else
            topology_analysis["status"] if topology_fault else
            "COUNT_MISMATCH" if count_mismatch and not topology_pending else
            "TOPOLOGY_PENDING" if topology_pending else "OK")
        results.append({"comm_id": comm_id, "counter": counter, "status": status,
                        "ranks": ranks, "expected_ranks": list(range(expected)),
                        "missing_ranks": missing, "incomplete_ranks": incomplete,
                        "slow_ranks": slow_ranks, "send_counts": send_values,
                        "recv_counts": recv_values, "per_rank": dict(by_rank),
                        "op_counts": op_counts,
                        "topology_edges": topology_edges,
                        "topology_groups": topology_groups,
                        "topology_ready": bool(topology_groups) and len(topology_groups) == len(rows),
                        "topology_analysis": topology_analysis,
                        "algorithms": algorithms, "protocols": protocols,
                        "nnodes": nnodes,
                        "records": len(rows)})
    return results

def analyze_completion_windows(records, *, window_size=5, slow_rate_ratio=0.5,
                               min_slow_windows=2):
    """Detect ranks that remain slow across several consecutive collectives.

    The per-collective detector is useful for an immediate alert, but a single
    noisy iteration should not be treated as a persistent communication issue.
    This function reuses the same device-clock rates and only compares ranks
    inside each collective, so it does not require clock synchronization across
    hosts.  It returns one result per communicator and is deliberately opt-in.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    groups = defaultdict(list)
    for row in records:
        if row.get("phase") == "complete":
            groups[str(row.get("comm_id", "unknown"))].append(row)
    results = []
    for comm_id, rows in sorted(groups.items()):
        counters = defaultdict(list)
        for row in rows:
            counters[str(row.get("counter", "unknown"))].append(row)
        def order(item):
            try:
                return (0, int(item[0]))
            except ValueError:
                return (1, item[0])
        per_collective = [(counter, analyze_completions(items,
                            slow_rate_ratio=slow_rate_ratio)[0])
                          for counter, items in sorted(counters.items(), key=order)]
        persistent = defaultdict(int)
        observed = 0
        for offset in range(0, max(0, len(per_collective) - window_size + 1)):
            window = per_collective[offset:offset + window_size]
            rate_maps = []
            for _, item in window:
                rates = {int(rank): value["send_rate"]
                         for rank, value in item["per_rank"].items()
                         if value["send_rate"] is not None and value["send_rate"] > 0}
                rate_maps.append(rates)
            common = set(rate_maps[0]) if rate_maps else set()
            for rates in rate_maps[1:]:
                common &= set(rates)
            if len(common) < 2:
                continue
            observed += 1
            for rank in common:
                slow_count = 0
                for rates in rate_maps:
                    baseline = median(rates.values())
                    if rates[rank] < baseline * slow_rate_ratio:
                        slow_count += 1
                if slow_count >= min_slow_windows:
                    persistent[rank] += 1
        slow_ranks = sorted(rank for rank, count in persistent.items() if count)
        if observed and slow_ranks:
            results.append({"comm_id": comm_id, "status": "COMM_SLOW_WINDOW",
                            "slow_ranks": slow_ranks,
                            "window_size": window_size,
                            "observed_windows": observed,
                            "persistent_windows": dict(persistent)})
    return results
