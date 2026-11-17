#!/usr/bin/env python3
"""Group CCL-D records and detect collective mismatches and rank gaps."""
import json
import pathlib
import time
from collections import defaultdict
from statistics import median

FIELDS = ("op", "count", "bytes", "datatype", "root")

def _root_cause(status, *, ranks=None, missing_ranks=None, incomplete_ranks=None,
                slow_ranks=None, mismatches=None, topology_analysis=None,
                per_rank=None, send_counts=None, recv_counts=None,
                peer_data_truncated_ranks=None):
    """Attach a stable, machine-readable diagnosis to every result.

    ``status`` remains the compact compatibility field consumed by existing
    scripts.  The additional fields make the monitor output actionable: a
    caller can identify the first affected rank and the evidence used by the
    decision without reimplementing the analyzer's precedence rules.
    """
    ranks = sorted(ranks or [])
    missing_ranks = sorted(missing_ranks or [])
    incomplete_ranks = sorted(incomplete_ranks or [])
    slow_ranks = sorted(slow_ranks or [])
    if status == "MISMATCH":
        cause, confidence, affected = "PARAMETER_MISMATCH", "high", ranks
    elif status == "MISSING_RANK":
        cause, confidence, affected = "RANK_DID_NOT_ENQUEUE", "high", missing_ranks
    elif status == "HANG_MISSING_RANK":
        cause, confidence, affected = "RANK_DID_NOT_ENTER_COLLECTIVE", "high", missing_ranks
    elif status == "HANG_PARTIAL_COMPLETION":
        cause, confidence, affected = "PARTIAL_RANKS_STUCK_IN_COLLECTIVE", "high", incomplete_ranks
    elif status == "HANG_ALL_RANKS":
        cause, confidence, affected = "ALL_RANKS_STUCK_IN_COLLECTIVE", "high", incomplete_ranks
    elif status in ("COMM_SLOW", "COMM_SLOW_WINDOW", "COMM_STALLED"):
        cause, confidence, affected = "COMMUNICATION_PROGRESS_SLOW", "medium", slow_ranks
    elif status == "COMPUTATION_SLOW":
        cause, confidence, affected = "COMPUTATION_STEP_SLOW", "medium", slow_ranks
    elif status == "MIXED_SLOW":
        cause, confidence, affected = "COMPUTATION_AND_COMMUNICATION_SLOW", "medium", slow_ranks
    elif status in ("RING_ASYMMETRY", "TREE_ASYMMETRY"):
        cause, confidence, affected = "TOPOLOGY_PROGRESS_ASYMMETRY", "medium", ranks
    elif status == "COUNT_MISMATCH":
        cause, confidence, affected = "DEVICE_TRANSFER_COUNT_MISMATCH", "medium", ranks
    elif status == "ENQUEUE_SKEW":
        cause, confidence, affected = "HOST_ENQUEUE_SKEW", "low", ranks
    elif status == "TOPOLOGY_PENDING":
        cause, confidence, affected = "INSUFFICIENT_TOPOLOGY_EVIDENCE", "low", ranks
    elif status == "OBSERVATION_DROPPED":
        cause, confidence, affected = "DIAGNOSTIC_RESOURCE_EXHAUSTED", "low", ranks
    elif status == "PENDING":
        cause, confidence, affected = "WAITING_FOR_RANK_RECORDS", "low", missing_ranks
    else:
        cause, confidence, affected = "NONE", "high", []
    evidence = {}
    if mismatches:
        evidence["mismatches"] = mismatches
    if missing_ranks:
        evidence["missing_ranks"] = missing_ranks
    if incomplete_ranks:
        evidence["incomplete_ranks"] = incomplete_ranks
    if slow_ranks:
        evidence["slow_ranks"] = slow_ranks
    if send_counts is not None:
        evidence["send_counts"] = send_counts
    if recv_counts is not None:
        evidence["recv_counts"] = recv_counts
    if topology_analysis and topology_analysis.get("status") not in (None, "NO_TOPOLOGY"):
        evidence["topology"] = topology_analysis
    if peer_data_truncated_ranks:
        evidence["peer_data_truncated_ranks"] = sorted(peer_data_truncated_ranks)
    if per_rank and slow_ranks:
        evidence["rates"] = {str(rank): {"send_rate": per_rank[rank].get("send_rate"),
                                          "recv_rate": per_rank[rank].get("recv_rate")}
                              for rank in slow_ranks if rank in per_rank}
    return {"root_cause": cause, "confidence": confidence,
            "first_error_rank": min(affected) if affected else None,
            "evidence": evidence}

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
                    if record.get("phase") in ("enqueue", "complete", "sample"):
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
        dropped = [row for row in rows
                   if row.get("diagnostic_status") == "OBSERVATION_DROPPED"]
        if dropped:
            ranks = sorted({int(row.get("rank", -1)) for row in rows})
            dropped_ranks = sorted({int(row.get("rank", -1)) for row in dropped})
            expected = max((int(row.get("nranks", 1)) for row in rows), default=1)
            missing = sorted(set(range(expected)) - set(ranks))
            p2p_rows = all(str(row.get("op", "")) in ("Send", "Recv") for row in rows)
            compare_fields = ("datatype",) if p2p_rows else FIELDS
            mismatches = {field: sorted({str(row.get(field)) for row in rows})
                          for field in compare_fields
                          if len({str(row.get(field)) for row in rows}) > 1}
            if mismatches:
                status = "MISMATCH"
            elif missing:
                status = "PENDING"
            else:
                status = "OBSERVATION_DROPPED"
            result = {"comm_id": comm_id, "counter": counter,
                      "trace_id": rows[0].get("trace_id"),
                      "status": status, "ranks": ranks,
                      "expected_ranks": list(range(expected)),
                      "missing_ranks": missing, "mismatches": mismatches,
                      "nnodes": max((int(row.get("nnodes", 1)) for row in rows), default=1),
                      "records": len(rows), "p2p": bool(p2p_rows),
                      "dropped_ranks": dropped_ranks}
            result.update(_root_cause(status, ranks=dropped_ranks if status == "OBSERVATION_DROPPED" else ranks,
                                      missing_ranks=missing, mismatches=mismatches))
            results.append(result)
            continue
        ranks = sorted({int(row.get("rank", -1)) for row in rows})
        expected = max((int(row.get("nranks", 1)) for row in rows), default=1)
        nnodes = max((int(row.get("nnodes", 1)) for row in rows), default=1)
        nnodes = max((int(row.get("nnodes", 1)) for row in rows), default=1)
        expected_ranks = list(range(expected))
        missing = sorted(set(expected_ranks) - set(ranks))
        p2p_rows = rows and all(str(row.get("op", "")) in ("Send", "Recv") for row in rows)
        # P2P groups may legitimately contain different peers and message
        # sizes. Their contract is checked edge-by-edge below.
        compare_fields = ("datatype",) if p2p_rows else FIELDS
        mismatches = {field: sorted({str(row.get(field)) for row in rows})
                      for field in compare_fields if len({str(row.get(field)) for row in rows}) > 1}
        times = [int(row["timestamp_ns"]) for row in rows if row.get("timestamp_ns") is not None]
        age_ms = max(0, (now_ns - min(times)) / 1_000_000) if times else 0
        skew_ms = (max(times) - min(times)) / 1_000_000 if len(times) > 1 else 0

        if mismatches:
            status = "MISMATCH"
        elif missing and not p2p_rows and age_ms >= timeout_ms:
            status = "MISSING_RANK"
        elif missing and not p2p_rows:
            status = "PENDING"
        elif nnodes <= 1 and skew_ms >= slow_skew_ms:
            # This is only host enqueue timing.  It is not a send/recv-rate
            # measurement and must not be presented as a communication slow diagnosis.
            status = "ENQUEUE_SKEW"
        else:
            status = "OK"
        result = {"comm_id": comm_id, "counter": counter,
                        "trace_id": rows[0].get("trace_id"), "status": status,
                        "ranks": ranks, "expected_ranks": expected_ranks,
                        "missing_ranks": missing, "mismatches": mismatches,
                        "nnodes": nnodes, "skew_ms": round(skew_ms, 3),
                        "age_ms": round(age_ms, 1),
                        "records": len(rows), "p2p": bool(p2p_rows)}
        result.update(_root_cause(status, ranks=ranks, missing_ranks=missing,
                                  mismatches=mismatches))
        results.append(result)
    return results

def analyze_operation_semantics(records):
    """Apply operation-specific contracts to enqueue and completion evidence.

    The generic analyzer checks that ranks entered the same operation.  This
    layer checks the communication contract: P2P edges are reciprocal,
    Broadcast has one root, and AllToAll has a complete rank-to-rank matrix.
    It intentionally consumes existing enqueue fields, so changing these
    rules does not require rebuilding RCCL.
    """
    enqueues = [r for r in records if r.get("phase") == "enqueue"]
    groups = defaultdict(list)
    for row in enqueues:
        comm_id = str(row.get("comm_id", "unknown"))
        op = str(row.get("op", ""))
        if op in ("Send", "Recv") and row.get("op_count") is not None:
            # RCCL gives each API call in a grouped P2P launch its own CCL-D
            # counter, while the group shares one opCount. Pair the send/recv
            # contract at that group boundary, not by the per-call TraceID.
            key = (comm_id, "p2p-op-count", str(row["op_count"]))
        else:
            key = (comm_id, "trace", str(row.get("counter", row.get("trace_id", "unknown"))))
        groups[key].append(row)
    results = []
    for group_key, rows in sorted(groups.items()):
        comm_id = group_key[0]
        counter = group_key[-1]
        ops = sorted({str(r.get("op", "")) for r in rows})
        if not ops:
            continue
        op = ops[0] if len(ops) == 1 else "MIXED"
        expected = max((int(r.get("nranks", 1)) for r in rows), default=1)
        status = "OK"
        evidence = {"op": op, "records": len(rows)}
        if op in ("Send", "Recv") or (op == "MIXED" and set(ops) <= {"Send", "Recv"}):
            sends = defaultdict(list)
            recvs = defaultdict(list)
            for r in rows:
                key = (int(r.get("rank", -1)), int(r.get("root", -1)))
                (sends if r.get("op") == "Send" else recvs)[key].append(r)
            missing_send = sorted((rank, peer) for (rank, peer) in sends
                                  if (peer, rank) not in recvs)
            missing_recv = sorted((rank, peer) for (rank, peer) in recvs
                                  if (peer, rank) not in sends)
            edge_mismatch = []
            for (rank, peer), send_items in sends.items():
                recv_items = recvs.get((peer, rank))
                if recv_items is not None:
                    send_bytes = sum(int(x.get("bytes", 0)) for x in send_items)
                    recv_bytes = sum(int(x.get("bytes", 0)) for x in recv_items)
                    send_count = sum(int(x.get("count", 0)) for x in send_items)
                    recv_count = sum(int(x.get("count", 0)) for x in recv_items)
                    send_types = sorted({str(x.get("datatype")) for x in send_items})
                    recv_types = sorted({str(x.get("datatype")) for x in recv_items})
                    diff = {}
                    if send_bytes != recv_bytes: diff["bytes"] = [send_bytes, recv_bytes]
                    if send_count != recv_count: diff["count"] = [send_count, recv_count]
                    if send_types != recv_types: diff["datatype"] = [send_types, recv_types]
                    if diff:
                        edge_mismatch.append({"send": [rank, peer], "recv": [peer, rank],
                                              "fields": diff})
            if missing_send or missing_recv:
                status = "P2P_PEER_MISSING"
            elif edge_mismatch:
                status = "P2P_TRANSFER_MISMATCH"
            evidence.update({"missing_send_edges": missing_send,
                             "missing_recv_edges": missing_recv,
                             "edge_mismatches": edge_mismatch})
        elif op == "Broadcast":
            roots = sorted({int(r.get("root", -1)) for r in rows})
            invalid = [root for root in roots if root < 0 or root >= expected]
            if len(roots) != 1 or invalid or len({int(r.get("rank", -1)) for r in rows}) != expected:
                status = "BROADCAST_ROLE_MISMATCH"
            evidence.update({"roots": roots, "invalid_roots": invalid,
                             "expected_root": roots[0] if len(roots) == 1 else None})
        elif op == "AllToAll":
            ranks = {int(r.get("rank", -1)) for r in rows}
            if ranks != set(range(expected)):
                status = "ALLTOALL_RANK_MISSING"
            shapes = {(str(r.get("count")), str(r.get("bytes")), str(r.get("datatype"))) for r in rows}
            if len(shapes) > 1:
                status = "ALLTOALL_SHAPE_MISMATCH"
            evidence.update({"ranks": sorted(ranks), "shape_variants": sorted(shapes)})
        elif op in ("AllGather", "ReduceScatter", "AllReduce"):
            shapes = {(str(r.get("count")), str(r.get("bytes")), str(r.get("datatype"))) for r in rows}
            if len(shapes) > 1:
                status = "COLLECTIVE_SHAPE_MISMATCH"
            evidence["shape_variants"] = sorted(shapes)
        if op == "MIXED" and set(ops) <= {"Send", "Recv"}:
            op = "P2P"
            evidence["communication_pattern"] = (
                "all-to-all-like" if len({(int(r.get("rank", -1)), int(r.get("root", -1))) for r in rows})
                >= expected * expected else "point-to-point")
        root = _root_cause("MISMATCH" if status != "OK" else "OK",
                           ranks=sorted({int(r.get("rank", -1)) for r in rows}),
                           mismatches=evidence)
        result = {"comm_id": comm_id, "counter": counter,
                  "status": status, "operation": op, "evidence": evidence,
                  "root_cause": root["root_cause"], "confidence": root["confidence"],
                  "first_error_rank": root["first_error_rank"]}
        if group_key[1] == "p2p-op-count":
            result["grouping"] = "op_count"
        if status != "OK":
            result["root_cause"] = {"P2P_PEER_MISSING": "P2P_PEER_RELATION_MISMATCH",
                                     "P2P_TRANSFER_MISMATCH": "P2P_TRANSFER_SIZE_MISMATCH",
                                     "BROADCAST_ROLE_MISMATCH": "BROADCAST_ROOT_MISMATCH",
                                     "ALLTOALL_RANK_MISSING": "ALLTOALL_RANK_SET_MISMATCH",
                                     "ALLTOALL_SHAPE_MISMATCH": "ALLTOALL_SHAPE_MISMATCH",
                                     "COLLECTIVE_SHAPE_MISMATCH": "COLLECTIVE_SHAPE_MISMATCH"}[status]
            result["confidence"] = "high"
        results.append(result)
    return results

def analyze_completions(records, *, slow_rate_ratio=0.5, now_ns=None,
                        timeout_ms=2000, step_skew_ms=2):
    """Aggregate completion records and detect missing completion/count/rate outliers.

    ``start_timestamp`` and ``end_timestamp`` are device clock ticks emitted by
    RCCL.  We only compare rates inside one collective, so no cross-node clock
    synchronization is required.  A slow-rate result is advisory and is only
    emitted when every participating rank has a usable duration.
    """
    now_mono_ns = time.monotonic_ns() if now_ns is None else now_ns
    now_wall_ns = time.time_ns() if now_ns is None else now_ns
    groups = defaultdict(list)
    enqueues = defaultdict(list)
    sample_groups = defaultdict(lambda: defaultdict(dict))
    dropped_keys = {(str(row.get("comm_id", "unknown")),
                     str(row.get("counter", "unknown")))
                    for row in records if row.get("phase") == "enqueue" and
                    row.get("diagnostic_status") == "OBSERVATION_DROPPED"}
    for record in records:
        if record.get("phase") == "enqueue":
            key = (str(record.get("comm_id", "unknown")),
                   str(record.get("counter", "unknown")))
            if key in dropped_keys:
                continue
            if record.get("op") in ("Send", "Recv"):
                continue
            enqueues[key].append(record)
            continue
        if record.get("phase") == "sample":
            key = (str(record.get("comm_id", "unknown")),
                   str(record.get("counter", "unknown")))
            rank = int(record.get("rank", -1))
            timestamp = int(record.get("timestamp_ns", 0))
            previous = sample_groups[key].setdefault(rank, {
                "timestamp_ns": timestamp, "send_delta": 0,
                "recv_delta": 0, "elapsed_ns": 0, "samples": 0})
            previous["timestamp_ns"] = max(previous["timestamp_ns"], timestamp)
            previous["send_delta"] += int(record.get("send_delta", 0))
            previous["recv_delta"] += int(record.get("recv_delta", 0))
            previous["elapsed_ns"] += max(0, int(record.get("elapsed_ns", 0)))
            previous["samples"] += 1
            continue
        # Records without CCL-D algorithm/protocol fields are not diagnostic completions.
        # It has topology_kind but no CCL-D algorithm/protocol fields and must
        # never be merged with the diagnostic slot completion.
        if (record.get("phase") != "complete" or record.get("op") in ("Send", "Recv") or
                ("algorithm" not in record and "topology_kind" in record)):
            continue
        key = (str(record.get("comm_id", "unknown")),
               str(record.get("counter", "unknown")))
        if key in dropped_keys:
            continue
        groups[key].append(record)
    for key in enqueues:
        groups.setdefault(key, [])
    results = []
    for (comm_id, counter), rows in sorted(groups.items()):
        enqueue_rows = enqueues.get((comm_id, counter), [])
        if not rows:
            expected = max((int(row.get("nranks", 1)) for row in enqueue_rows), default=1)
            enqueued = sorted({int(row.get("rank", -1)) for row in enqueue_rows})
            missing = sorted(set(range(expected)) - set(enqueued))
            ages = []
            for row in enqueue_rows:
                wall = row.get("timestamp_wall_ns")
                mono = row.get("timestamp_ns")
                if wall is not None:
                    ages.append(now_wall_ns - int(wall))
                elif mono is not None and int(row.get("nnodes", 1)) == 1:
                    ages.append(now_mono_ns - int(mono))
            age_ms = max(0, min(ages) / 1_000_000) if ages else 0
            if age_ms < timeout_ms:
                status = "PENDING"
            elif missing:
                status = "HANG_MISSING_RANK"
            elif enqueued:
                status = "HANG_ALL_RANKS"
            else:
                status = "PENDING"
            result = {"comm_id": comm_id, "counter": counter, "status": status,
                      "ranks": enqueued, "expected_ranks": list(range(expected)),
                      "missing_ranks": missing, "incomplete_ranks": enqueued,
                      "slow_ranks": [], "records": 0, "age_ms": round(age_ms, 1)}
            result.update(_root_cause(status, ranks=enqueued,
                                      missing_ranks=missing,
                                      incomplete_ranks=enqueued))
            results.append(result)
            continue
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
        expected = max((int(row.get("nranks", 1)) for row in rows + enqueue_rows), default=1)
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
        topology_count_fault_ranks = set()
        topology_compare_groups = []
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
                pending = []
                for channel, channel_rows in by_channel.items():
                    rank_rows = defaultdict(list)
                    for row in channel_rows:
                        rank_rows[int(row["rank"])].append(row)
                    by_rank_row = {}
                    bad = False
                    for rank, batch_rows in rank_rows.items():
                        topology = {(int(r["ring_prev"]), int(r["ring_next"])) for r in batch_rows}
                        if len(topology) != 1:
                            bad = True
                        by_rank_row[rank] = next(iter(topology))
                    ring_ranks = set(by_rank_row)
                    expected_ring_ranks = set(range(max((int(r.get("nranks", 0)) for r in channel_rows), default=0)))
                    if expected_ring_ranks and ring_ranks != expected_ring_ranks:
                        # A plan may expose only a subset of channels/ranks
                        # while another plan for the same TraceID is still
                        # being consumed. Do not call that partial evidence a
                        # broken ring; report it as pending topology evidence.
                        pending.append(channel)
                        continue
                    if any(prev not in ring_ranks or nxt not in ring_ranks
                           for prev, nxt in by_rank_row.values()):
                        bad = True
                    if not bad:
                        bad = any(by_rank_row[nxt][0] != rank or by_rank_row[prev][1] != rank
                                  for rank, (prev, nxt) in by_rank_row.items())
                    if bad:
                        asymmetric.append(channel)
                    else:
                        topology_compare_groups.append(sorted(by_rank_row))
                        counts = defaultdict(lambda: [0, 0])
                        for row in channel_rows:
                            counts[int(row["rank"])][0] += int(row.get("send_count", 0))
                            counts[int(row["rank"])][1] += int(row.get("recv_count", 0))
                        values = list(counts.values())
                        if values:
                            common = sorted(map(tuple, values))[len(values) // 2]
                            topology_count_fault_ranks.update(
                                rank for rank, value in counts.items() if tuple(value) != common)
                topology_status = ("RING_ASYMMETRY" if asymmetric else
                                   "RING_TOPOLOGY_PENDING" if pending else "RING_SYMMETRIC")
                topology_analysis = {"status": topology_status,
                                     "channels": sorted(by_channel),
                                     "asymmetric_channels": sorted(asymmetric),
                                     "pending_channels": sorted(pending)}
        elif tree_rows:
            required = ("tree_up", "tree_down0", "tree_down1", "tree_down2", "tree_depth")
            if any(any(r.get(field) is None for field in required) for r in tree_rows):
                topology_analysis = {"status": "TREE_LEVEL_PENDING",
                                     "reason": "completion records lack tree peers or depth"}
            else:
                # RCCL may export a kernel-wide tree depth value rather than
                # the logical level of each rank. Reconstruct levels from the
                # parent/children edges whenever the exported value is equal
                # for all ranks.
                tree_by_channel = defaultdict(dict)
                for row in tree_rows:
                    channel = int(row.get("channel", -1))
                    tree_by_channel[channel][int(row["rank"])] = row
                derived_depth = {}
                for channel, channel_rows in tree_by_channel.items():
                    if len({int(r["tree_depth"]) for r in channel_rows.values()}) == 1:
                        roots = [rank for rank, row in channel_rows.items()
                                 if int(row["tree_up"]) < 0]
                        if len(roots) == 1:
                            levels = {roots[0]: 0}
                            queue = [roots[0]]
                            while queue:
                                parent = queue.pop(0)
                                for field in ("tree_down0", "tree_down1", "tree_down2"):
                                    child = int(channel_rows[parent][field])
                                    if child >= 0 and child not in levels and child in channel_rows:
                                        levels[child] = levels[parent] + 1
                                        queue.append(child)
                            if len(levels) == len(channel_rows):
                                derived_depth[channel] = levels
                by_level = defaultdict(list)
                asymmetric = []
                for channel, channel_rows in tree_by_channel.items():
                    rank_rows = {int(rank): row for rank, row in channel_rows.items()}
                    roots = [rank for rank, row in rank_rows.items() if int(row["tree_up"]) < 0]
                    children = {rank: tuple(int(row[field]) for field in
                                ("tree_down0", "tree_down1", "tree_down2")
                                if int(row[field]) >= 0) for rank, row in rank_rows.items()}
                    invalid = len(roots) != 1
                    for rank, row in rank_rows.items():
                        parent = int(row["tree_up"])
                        if parent >= 0 and (parent not in rank_rows or rank not in children.get(parent, ())):
                            invalid = True
                        for child in children[rank]:
                            if child not in rank_rows or int(rank_rows[child]["tree_up"]) != rank:
                                invalid = True
                    levels = derived_depth.get(channel)
                    if levels is None:
                        levels = {rank: int(row["tree_depth"]) for rank, row in rank_rows.items()}
                    elif len(levels) != len(rank_rows):
                        invalid = True
                    if invalid:
                        asymmetric.append(channel)
                    for rank, row in rank_rows.items():
                        depth = levels.get(rank, int(row["tree_depth"]))
                        role = (int(int(row["tree_up"]) >= 0), len(children[rank]))
                        start, end = row.get("start_timestamp"), row.get("end_timestamp")
                        duration = int(end) - int(start) if start is not None and end is not None else 0
                        by_level[(channel, depth, role)].append({
                            "rank": rank,
                            "send_count": int(row.get("send_count", 0)),
                            "recv_count": int(row.get("recv_count", 0)),
                            "send_rate": (int(row.get("send_count", 0)) / duration
                                           if duration > 0 else None),
                            "recv_rate": (int(row.get("recv_count", 0)) / duration
                                           if duration > 0 else None)})
                for members in by_level.values():
                    topology_compare_groups.append(sorted(m["rank"] for m in members))
                    if len(members) < 2:
                        continue
                    values = [(m["send_count"], m["recv_count"]) for m in members]
                    common = sorted(values)[len(values) // 2]
                    topology_count_fault_ranks.update(
                        m["rank"] for m in members
                        if (m["send_count"], m["recv_count"]) != common)
                topology_analysis = {"status": "TREE_ASYMMETRY" if asymmetric else "TREE_SYMMETRIC",
                                     "channel_depths": [
                                         {"channel": c, "depth": d, "parent": role[0],
                                          "children": role[1], "ranks": len(members)}
                                         for (c, d, role), members in sorted(by_level.items())],
                                     "asymmetric_channels": sorted(asymmetric),
                                     "asymmetric_levels": []}
        # Ring ranks have symmetric degree. Tree and CollNet ranks may have
        # different degrees; those require topology-level grouping before a
        # count mismatch can be declared.
        count_mismatch = len(send_values) > 1 or len(recv_values) > 1
        operations = {str(row.get("op", "")) for row in rows + enqueue_rows}
        # Broadcast legitimately has a transmitting root and receiving peers;
        # aggregate send counts therefore differ by rank by design.
        if operations == {"Broadcast"}:
            count_mismatch = False
        topology_pending = count_mismatch and not topology_count_fault_ranks and (
            nnodes > 1 or any(a != -1 for a in algorithms))
        incomplete = sorted(rank for rank, value in by_rank.items() if not value["complete"])
        for value in by_rank.values():
            start = value["start_timestamp"]
            end = value["end_timestamp"]
            duration = end - start if start is not None and end is not None and end > start else None
            value["duration"] = duration
            value["send_rate"] = (value["send_count"] / duration) if duration else None
            value["recv_rate"] = (value["recv_count"] / duration) if duration else None
        # Slow diagnosis uses AntCCL's fixed-interval sample deltas. Completion
        # duration rates remain output as descriptive timing fields only.
        for rank, value in by_rank.items():
            sample = sample_groups.get((comm_id, counter), {}).get(rank)
            value["sample_send_delta"] = sample["send_delta"] if sample else None
            value["sample_recv_delta"] = sample["recv_delta"] if sample else None
            value["sample_interval_ns"] = (
                sample["elapsed_ns"] // max(1, sample["samples"])
                if sample else None)
        rate_send_key = "sample_send_delta"
        rate_recv_key = "sample_recv_delta"
        usable_send_rates = [v[rate_send_key] for v in by_rank.values()
                             if v[rate_send_key] is not None and v[rate_send_key] > 0]
        usable_recv_rates = [v[rate_recv_key] for v in by_rank.values()
                             if v[rate_recv_key] is not None and v[rate_recv_key] > 0]
        slow_ranks = []
        slow_diagnosis = None
        # A one-way zero can be a valid role/topology property (for example,
        # the broadcast root). Compare directions independently and diagnose
        # only when every participating rank has evidence in both directions.
        if not topology_compare_groups and not tree_rows and len(by_rank) >= 2:
            topology_compare_groups.append(sorted(by_rank))
        for group in topology_compare_groups:
            if len(group) < 2 or any(rank not in by_rank for rank in group):
                continue
            group_send = [by_rank[rank][rate_send_key] for rank in group]
            group_recv = [by_rank[rank][rate_recv_key] for rank in group]
            if any(value is None or value <= 0 for value in group_send + group_recv):
                continue
            send_baseline = sorted(group_send)[len(group_send) // 2]
            recv_baseline = sorted(group_recv)[len(group_recv) // 2]
            slow_ranks.extend(rank for rank in group
                              if by_rank[rank][rate_send_key] < send_baseline * slow_rate_ratio or
                              by_rank[rank][rate_recv_key] < recv_baseline * slow_rate_ratio)
        slow_ranks = sorted(set(slow_ranks))
        if slow_ranks:
            if slow_ranks:
                # AntCCL separates host/step desynchronization from a
                # communication-rate anomaly.  Device clocks are only
                # compared within one node; cross-node start timestamps are
                # deliberately excluded because they are not synchronized.
                same_node = nnodes <= 1
                starts = [v["start_timestamp"] for v in by_rank.values()
                          if v["start_timestamp"] is not None]
                start_skew = (max(starts) - min(starts)) if starts else 0
                compute_slow = same_node and start_skew >= int(step_skew_ms * 1_000_000)
                slow_diagnosis = ("MIXED" if compute_slow else "COMMUNICATION_SLOW")
        # AntCCL's first slow branch is step consistency.  Only use device
        # start timestamps when all ranks are on one node; cross-node clocks
        # are not a valid ordering source.  The shortest collective duration
        # identifies the compute-slow root in the documented case.
        start_values = [v["start_timestamp"] for v in by_rank.values()
                        if v["start_timestamp"] is not None]
        step_skew = (max(start_values) - min(start_values)) if start_values else 0
        compute_root = []
        if nnodes <= 1 and step_skew >= int(step_skew_ms * 1_000_000):
            durations = [(rank, v["duration"]) for rank, v in by_rank.items()
                         if v["duration"] is not None]
            if durations:
                minimum = min(value for _, value in durations)
                compute_root = sorted(rank for rank, value in durations
                                      if value == minimum)
            slow_diagnosis = ("MIXED" if slow_diagnosis in ("COMMUNICATION_SLOW", "MIXED")
                              else "COMPUTATION_SLOW")
        topology_fault = topology_analysis["status"] in ("RING_ASYMMETRY", "TREE_ASYMMETRY")
        complete_ranks = set(by_rank)
        enqueued_ranks = {int(row.get("rank", -1)) for row in enqueue_rows}
        incomplete.extend(sorted(enqueued_ranks - complete_ranks))
        incomplete = sorted(set(incomplete))
        missing_completion = sorted(enqueued_ranks - complete_ranks)
        # A rank absent from completion but present in enqueue entered the
        # collective and belongs to the partial/all-ranks hang branches. Only
        # an absent enqueue is the AntCCL "rank did not enter" branch.
        missing = (sorted(set(range(expected)) - enqueued_ranks)
                   if enqueue_rows else missing)
        diagnosed_slow = sorted(set(slow_ranks) | set(compute_root))
        status = ("HANG_MISSING_RANK" if missing else
            "HANG_ALL_RANKS" if incomplete and len(incomplete) == expected else
            "HANG_PARTIAL_COMPLETION" if incomplete else
            "MIXED_SLOW" if slow_diagnosis == "MIXED" else
            "COMPUTATION_SLOW" if slow_diagnosis == "COMPUTATION_SLOW" else
            "COUNT_MISMATCH" if (count_mismatch and not topology_pending) else
            "COMM_SLOW" if slow_ranks else
            topology_analysis["status"] if topology_fault else
            "TOPOLOGY_PENDING" if topology_pending else
            "OK")
        if (enqueue_rows and (missing or incomplete) and not incomplete):
            ages = []
            for row in enqueue_rows:
                wall = row.get("timestamp_wall_ns")
                mono = row.get("timestamp_ns")
                if wall is not None:
                    ages.append(now_wall_ns - int(wall))
                elif mono is not None and int(row.get("nnodes", 1)) == 1:
                    ages.append(now_mono_ns - int(mono))
            if not ages or min(ages) / 1_000_000 < timeout_ms:
                status = "PENDING"
        result = {"comm_id": comm_id, "counter": counter, "status": status,
                        "ranks": ranks, "expected_ranks": list(range(expected)),
                        "missing_ranks": missing, "incomplete_ranks": incomplete,
                        "slow_ranks": diagnosed_slow, "send_counts": send_values,
                        "recv_counts": recv_values, "per_rank": dict(by_rank),
                        "slow_diagnosis": slow_diagnosis,
                        "step_skew_ms": round(step_skew / 1_000_000, 3),
                        "op_counts": op_counts,
                        "topology_edges": topology_edges,
                        "topology_groups": topology_groups,
                        "count_mismatch_ranks": sorted(topology_count_fault_ranks),
                        "topology_ready": bool(topology_groups) and len(topology_groups) == len(rows),
                        "topology_analysis": topology_analysis,
                        "algorithms": algorithms, "protocols": protocols,
                        "nnodes": nnodes,
                        "records": len(rows)}
        result.update(_root_cause(status,
                                  ranks=(sorted(topology_count_fault_ranks)
                                         if status == "COUNT_MISMATCH" and topology_count_fault_ranks
                                         else ranks),
                                  missing_ranks=missing,
                                  incomplete_ranks=incomplete, slow_ranks=diagnosed_slow,
                                  topology_analysis=topology_analysis,
                                  per_rank=by_rank, send_counts=send_values,
                                  recv_counts=recv_values,
                                  peer_data_truncated_ranks=[]))
        results.append(result)
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
        if row.get("phase") in ("complete", "sample"):
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
                rates = {int(rank): value["sample_send_delta"]
                         for rank, value in item["per_rank"].items()
                         if value["sample_send_delta"] is not None}
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
            result = {"comm_id": comm_id, "status": "COMM_SLOW_WINDOW",
                            "slow_ranks": slow_ranks,
                            "window_size": window_size,
                            "observed_windows": observed,
                            "persistent_windows": dict(persistent)}
            result.update(_root_cause("COMM_SLOW_WINDOW", slow_ranks=slow_ranks))
            results.append(result)
    return results

def analyze_progress_samples(records, *, stagnant_intervals=3):
    """Find a rank with no counter progress while a same-node peer progresses.

    Samples use each host's monotonic clock. Cross-node timestamp comparison is
    intentionally avoided; this detector only compares ranks reported on the
    same node and within the same communicator/operation counter.
    """
    if stagnant_intervals < 2:
        raise ValueError("stagnant_intervals must be at least 2")
    completed = {(str(row.get("comm_id", "unknown")),
                  str(row.get("counter", "unknown")))
                 for row in records if row.get("phase") == "complete"}
    completed.update((str(row.get("comm_id", "unknown")),
                      str(row.get("counter", "unknown")))
                     for row in records if row.get("phase") == "enqueue" and
                     row.get("diagnostic_status") == "OBSERVATION_DROPPED")
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    expected_by_node = defaultdict(int)
    for row in records:
        if row.get("phase") != "sample" or row.get("timestamp_ns") is None:
            continue
        node = int(row.get("node", -1))
        rank = int(row.get("rank", -1))
        channel = int(row.get("channel", -1))
        key = (str(row.get("comm_id", "unknown")),
               str(row.get("counter", "unknown")))
        if key in completed:
            continue
        ts = int(row["timestamp_ns"])
        local_count = int(row.get("local_ranks", row.get("nranks", 1)))
        expected_by_node[(str(row.get("comm_id", "unknown")),
                          str(row.get("counter", "unknown")), node)] = max(
                              expected_by_node[(str(row.get("comm_id", "unknown")),
                                                 str(row.get("counter", "unknown")), node)],
                              local_count, int(row.get("local_rank", rank)) + 1)
        bucket = groups[(str(row.get("comm_id", "unknown")),
                         str(row.get("counter", "unknown")))][node][rank][channel]
        item = bucket.setdefault(ts, {"send_count": 0, "recv_count": 0,
                                      "counter_reset": False})
        item["send_count"] += int(row.get("send_count", 0))
        item["recv_count"] += int(row.get("recv_count", 0))
        item["counter_reset"] = item["counter_reset"] or bool(row.get("counter_reset", False))

    results = []
    for (comm_id, counter), nodes in sorted(groups.items()):
        stalled = []
        progress_by_node_rank = defaultdict(bool)
        for node, ranks in nodes.items():
            totals = {}
            made_progress = {}
            for rank, channels in ranks.items():
                for channel, samples in channels.items():
                    points = sorted(samples.items())
                    intervals = []
                    for (t0, first), (t1, second) in zip(points, points[1:]):
                        if first.get("counter_reset") or second.get("counter_reset"):
                            continue
                        intervals.append((t0, t1,
                                          second["send_count"] - first["send_count"],
                                          second["recv_count"] - first["recv_count"]))
                    totals[(rank, channel)] = intervals
                    started_with_progress = bool(points and (
                        points[0][1].get("send_count", 0) > 0 or
                        points[0][1].get("recv_count", 0) > 0))
                    made_progress[(rank, channel)] = started_with_progress or any(
                        send > 0 or recv > 0 for _, _, send, recv in intervals)
                    progress_by_node_rank[(node, rank)] = (
                        progress_by_node_rank[(node, rank)] or made_progress[(rank, channel)])
            for (rank, channel), intervals in totals.items():
                for offset in range(0, len(intervals) - stagnant_intervals + 1):
                    window = intervals[offset:offset + stagnant_intervals]
                    if any(send > 0 or recv > 0 for _, _, send, recv in window):
                        continue
                    start, end = window[0][0], window[-1][1]
                    peer_progress = []
                    for (peer, peer_channel), peer_intervals in totals.items():
                        if peer == rank or peer_channel != channel:
                            continue
                        if made_progress[(peer, peer_channel)] or any(
                                t1 >= start and t0 <= end and (send > 0 or recv > 0)
                                for t0, t1, send, recv in peer_intervals):
                            peer_progress.append(peer)
                    observed_ranks = {peer for peer, peer_channel in totals
                                      if peer_channel == channel}
                    expected = expected_by_node[(comm_id, counter, node)]
                    if peer_progress or (expected > 1 and len(observed_ranks) == expected):
                        stalled.append({"rank": rank, "node": node,
                                        "channel": channel,
                                        "start_ns": start, "end_ns": end,
                                        "intervals": stagnant_intervals,
                                        "duration_ms": round((end - start) / 1_000_000, 3),
                                        "progressing_peers": sorted(peer_progress),
                                        "node_ranks_observed": len(observed_ranks),
                                        "expected_ranks": expected})
                        break
        if stalled:
            # Prefer ranks that made no progress at all while local peers did.
            # Ranks that progressed and later waited at a collective boundary
            # are participants in the stall, not necessarily its source.
            no_progress = [item for item in stalled
                           if not progress_by_node_rank[(item["node"], item["rank"])] ]
            if no_progress:
                stalled = no_progress
            slow_ranks = sorted({item["rank"] for item in stalled})
            all_ranks_stalled = bool(expected_by_node) and all(
                len(nodes.get(node, {})) == expected and
                all(not progress_by_node_rank[(node, rank)]
                    for rank in nodes.get(node, {}))
                for (comm, trace, node), expected in expected_by_node.items()
                if comm == comm_id and trace == counter and expected > 1)
            result = {"comm_id": comm_id, "counter": counter,
                      "status": "COMM_STALLED", "slow_ranks": slow_ranks,
                      "stalled_intervals": stalled,
                      "scope": "all_ranks" if all_ranks_stalled else "rank_progress"}
            result.update(_root_cause("COMM_STALLED", slow_ranks=slow_ranks))
            result["root_cause"] = ("ALL_RANKS_NO_PROGRESS" if all_ranks_stalled
                                    else "RANK_PROGRESS_STALLED")
            result["evidence"]["stalled_intervals"] = stalled
            results.append(result)
    return results

def analyze_non_collective_hang(records):
    """Report an externally marked hang after collective completion.

    The communication trace cannot attribute a post-collective hang to a rank;
    an explicit application/monitor event is therefore surfaced as an
    unlocalized non-communication hang instead of inventing a communication
    root cause.
    """
    results = []
    for row in records:
        if row.get("phase") != "hang" or row.get("scope") not in ("non_collective", "non_communication"):
            continue
        result = {"comm_id": str(row.get("comm_id", "unknown")),
                  "counter": str(row.get("counter", "unknown")),
                  "status": "NON_COMMUNICATION_HANG_UNLOCALIZED",
                  "ranks": sorted(row.get("ranks", [])),
                  "evidence": {"event": row}}
        result.update(_root_cause("NON_COMMUNICATION_HANG_UNLOCALIZED"))
        result["root_cause"] = "NON_COMMUNICATION_HANG_UNLOCALIZED"
        result["confidence"] = "high"
        results.append(result)
    return results
