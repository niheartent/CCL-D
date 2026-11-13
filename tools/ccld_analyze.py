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
    elif status == "MISSING_COMPLETION":
        cause, confidence, affected = "COLLECTIVE_DID_NOT_COMPLETE", "high", incomplete_ranks or missing_ranks
    elif status in ("COMM_SLOW", "COMM_SLOW_WINDOW", "COMM_STALLED"):
        cause, confidence, affected = "COMMUNICATION_PROGRESS_SLOW", "medium", slow_ranks
    elif status in ("RING_ASYMMETRY", "TREE_ASYMMETRY"):
        cause, confidence, affected = "TOPOLOGY_PROGRESS_ASYMMETRY", "medium", ranks
    elif status == "COUNT_MISMATCH":
        cause, confidence, affected = "DEVICE_TRANSFER_COUNT_MISMATCH", "medium", ranks
    elif status == "ENQUEUE_SKEW":
        cause, confidence, affected = "HOST_ENQUEUE_SKEW", "low", ranks
    elif status == "TOPOLOGY_PENDING":
        cause, confidence, affected = "INSUFFICIENT_TOPOLOGY_EVIDENCE", "low", ranks
    elif status == "PEER_DATA_TRUNCATED":
        cause, confidence, affected = "PEER_FLOW_EVIDENCE_TRUNCATED", "low", peer_data_truncated_ranks or ranks
    elif status == "PEER_DATA_INCONSISTENT":
        cause, confidence, affected = "PEER_FLOW_EVIDENCE_INCONSISTENT", "low", ranks
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

def analyze_completions(records, *, slow_rate_ratio=0.5):
    """Aggregate completion records and detect missing completion/count/rate outliers.

    ``start_timestamp`` and ``end_timestamp`` are device clock ticks emitted by
    RCCL.  We only compare rates inside one collective, so no cross-node clock
    synchronization is required.  A slow-rate result is advisory and is only
    emitted when every participating rank has a usable duration.
    """
    groups = defaultdict(list)
    for record in records:
        # ENABLE_COLLTRACE writes a separate legacy completion-shaped event.
        # It has topology_kind but no CCL-D algorithm/protocol fields and must
        # never be merged with the diagnostic slot completion.
        if (record.get("phase") != "complete" or
                ("algorithm" not in record and "topology_kind" in record)):
            continue
        # P2P work has separate sendOpCount/recvOpCount and can be present on
        # only a subset of ranks in a channel batch. Do not treat it as one
        # rank-synchronous collective completion.
        if int(record.get("algorithm", -1)) == 255:
            continue
        key = (str(record.get("comm_id", "unknown")),
               str(record.get("counter", "unknown")))
        groups[key].append(record)
    results = []
    for (comm_id, counter), rows in sorted(groups.items()):
        by_rank = defaultdict(lambda: {"send_count": 0, "recv_count": 0,
                                       "send_count_by_protocol": [0, 0, 0],
                                       "recv_count_by_protocol": [0, 0, 0],
                                       "send_peers": {}, "recv_peers": {},
                                       "send_protocol_peers": {}, "recv_protocol_peers": {},
                                       "peer_overflow": False,
                                       "dropped_batches": 0,
                                       "peer_data_inconsistent": [],
                                       "channels": 0, "complete": True,
                                       "start_timestamp": None,
                                       "end_timestamp": None})
        for row in rows:
            rank = int(row.get("rank", -1))
            item = by_rank[rank]
            item["send_count"] += int(row.get("send_count", 0))
            item["recv_count"] += int(row.get("recv_count", 0))
            for field in ("send_count_by_protocol", "recv_count_by_protocol"):
                values = row.get(field)
                if isinstance(values, list) and len(values) == 3:
                    item[field] = [a + int(b) for a, b in zip(item[field], values)]
            for field in ("send_peers", "recv_peers"):
                for edge in row.get(field, []) or []:
                    if not isinstance(edge, dict) or edge.get("peer") is None:
                        continue
                    peer = str(int(edge["peer"]))
                    item[field][peer] = item[field].get(peer, 0) + int(edge.get("bytes", 0))
            for field, peer_field in (("send_peer_protocol_bytes", "send_peers"),
                                      ("recv_peer_protocol_bytes", "recv_peers")):
                matrix = row.get(field)
                peers = row.get(peer_field) or []
                if isinstance(matrix, list):
                    for proto, values in enumerate(matrix[:3]):
                        proto_key = str(proto)
                        proto_map = item[field.replace("_peer_protocol_bytes", "_protocol_peers")].setdefault(proto_key, {})
                        for peer_index, value in enumerate(values or []):
                            if peer_index >= len(peers):
                                continue
                            peer_key = str(int(peers[peer_index]["peer"]))
                            proto_map[peer_key] = proto_map.get(peer_key, 0) + int(value)
            item["peer_overflow"] = item["peer_overflow"] or bool(row.get("peer_overflow", False))
            item["dropped_batches"] += int(row.get("dropped_batches", 0))
            channel = int(row.get("channel", -1))
            for field, peer_field, count_field in (
                    ("send_peer_protocol_bytes", "send_peers", "send_count_by_protocol"),
                    ("recv_peer_protocol_bytes", "recv_peers", "recv_count_by_protocol")):
                matrix = row.get(field)
                counts = row.get(count_field)
                if not isinstance(matrix, list) or not isinstance(counts, list) or len(matrix) != 3 or len(counts) != 3:
                    continue
                observed = [sum(int(x) for x in values or []) for values in matrix]
                expected = [int(x) for x in counts]
                if observed != expected:
                    item["peer_data_inconsistent"].append({
                        "channel": channel, "direction": field.split("_")[0],
                        "expected_by_protocol": expected,
                        "observed_by_protocol": observed})
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
        dropped = sorted(rank for rank, value in by_rank.items() if value["dropped_batches"] > 0)
        status = "DIAGNOSTIC_BATCH_DROPPED" if dropped else (
            "MISSING_COMPLETION" if missing or incomplete else
            "COMM_SLOW" if slow_ranks else
            topology_analysis["status"] if topology_fault else
            "COUNT_MISMATCH" if count_mismatch and not topology_pending else
            "TOPOLOGY_PENDING" if topology_pending else
            "PEER_DATA_TRUNCATED" if any(v["peer_overflow"] for v in by_rank.values()) else "OK")
        peer_inconsistent = []
        peer_inconsistent_channels = {}
        for rank, value in by_rank.items():
            if value["peer_data_inconsistent"]:
                peer_inconsistent.append(rank)
                peer_inconsistent_channels[str(rank)] = value["peer_data_inconsistent"]
            peer_send = sum(value["send_peers"].values())
            peer_recv = sum(value["recv_peers"].values())
            if peer_send > value["send_count"] or peer_recv > value["recv_count"]:
                peer_inconsistent.append(rank)
            for proto, total in enumerate(value["send_count_by_protocol"]):
                matrix = sum(int(x) for x in value["send_protocol_peers"].get(str(proto), {}).values())
                if matrix != total and (matrix or total):
                    peer_inconsistent.append(rank)
            for proto, total in enumerate(value["recv_count_by_protocol"]):
                matrix = sum(int(x) for x in value["recv_protocol_peers"].get(str(proto), {}).values())
                if matrix != total and (matrix or total):
                    peer_inconsistent.append(rank)
        if peer_inconsistent and status in ("OK", "TOPOLOGY_PENDING"):
            status = "PEER_DATA_INCONSISTENT"
        result = {"comm_id": comm_id, "counter": counter, "status": status,
                        "ranks": ranks, "expected_ranks": list(range(expected)),
                        "missing_ranks": missing, "incomplete_ranks": incomplete,
                        "slow_ranks": slow_ranks, "send_counts": send_values,
                        "recv_counts": recv_values, "per_rank": dict(by_rank),
                        "send_counts_by_protocol": [sum(v["send_count_by_protocol"][i] for v in by_rank.values()) for i in range(3)],
                        "recv_counts_by_protocol": [sum(v["recv_count_by_protocol"][i] for v in by_rank.values()) for i in range(3)],
                        "send_bytes_by_peer": {str(rank): v["send_peers"] for rank, v in by_rank.items()},
                        "recv_bytes_by_peer": {str(rank): v["recv_peers"] for rank, v in by_rank.items()},
                        "send_protocol_bytes_by_peer": {str(rank): v["send_protocol_peers"] for rank, v in by_rank.items()},
                        "recv_protocol_bytes_by_peer": {str(rank): v["recv_protocol_peers"] for rank, v in by_rank.items()},
                        "peer_data_truncated_ranks": sorted(rank for rank, value in by_rank.items()
                                                             if value["peer_overflow"]),
                        "peer_data_inconsistent_ranks": sorted(set(peer_inconsistent)),
                        "peer_data_inconsistent_channels": peer_inconsistent_channels,
                        "diagnostic_dropped_ranks": dropped,
                        "op_counts": op_counts,
                        "topology_edges": topology_edges,
                        "topology_groups": topology_groups,
                        "topology_ready": bool(topology_groups) and len(topology_groups) == len(rows),
                        "topology_analysis": topology_analysis,
                        "algorithms": algorithms, "protocols": protocols,
                        "nnodes": nnodes,
                        "records": len(rows)}
        result.update(_root_cause(status, ranks=ranks, missing_ranks=missing,
                                  incomplete_ranks=incomplete, slow_ranks=slow_ranks,
                                  topology_analysis=topology_analysis,
                                  per_rank=by_rank, send_counts=send_values,
                                  recv_counts=recv_values,
                                  peer_data_truncated_ranks=result["peer_data_truncated_ranks"]))
        results.append(result)
    return results

def analyze_p2p_completions(records):
    """Summarize P2P device peer evidence under its host P2P group.

    Send and Recv API calls have separate TraceIDs. When RCCL groups them into
    one launch, ``op_count`` is the shared host grouping key; device completion
    rows can therefore be associated with either API counter without claiming
    that the counters themselves are equal. AllToAll completions share the
    parent AllToAll counter and are excluded here because that parent is checked
    as a collective by ``analyze_operation_semantics``.
    """
    enqueue_by_counter = {}
    p2p_groups = defaultdict(list)
    for row in records:
        if row.get("phase") != "enqueue":
            continue
        comm_id = str(row.get("comm_id", "unknown"))
        counter = str(row.get("counter", "unknown"))
        enqueue_by_counter[(comm_id, counter, int(row.get("rank", -1)))] = row
        if row.get("op") in ("Send", "Recv"):
            group_id = str(row.get("op_count", counter))
            p2p_groups[(comm_id, group_id)].append(row)
    groups = defaultdict(list)
    for row in records:
        if row.get("phase") != "complete" or row.get("algorithm") != 255:
            continue
        comm_id = str(row.get("comm_id", "unknown"))
        rank = int(row.get("rank", -1))
        counter = str(row.get("counter", "unknown"))
        enqueue = enqueue_by_counter.get((comm_id, counter, rank))
        related = row.get("related_counter")
        related_enqueue = (enqueue_by_counter.get((comm_id, str(related), rank))
                           if related is not None else None)
        owners = [item for item in (enqueue, related_enqueue)
                  if item is not None and item.get("op") in ("Send", "Recv")]
        if not owners:
            # Device records belonging to AllToAll are already represented by
            # its parent TraceID; ignore those instead of calling them orphaned.
            if enqueue and enqueue.get("op") in ("AllToAll", "AllToAllPivot"):
                continue
            groups[(comm_id, "unassociated")].append(row)
            continue
        group_ids = {str(item.get("op_count", item.get("counter", "unknown")))
                     for item in owners}
        for group_id in group_ids:
            groups[(comm_id, group_id)].append(row)
    results = []
    for (comm_id, group_id), rows in sorted(groups.items()):
        if group_id == "unassociated":
            status = "P2P_COMPLETION_UNCORRELATED"
        else:
            status = "P2P_DEVICE_EVIDENCE"
        edges = defaultdict(lambda: {"send_bytes": 0, "recv_bytes": 0})
        ranks = set()
        for row in rows:
            rank = int(row.get("rank", -1)); ranks.add(rank)
            for field, direction, peer_field in (("send_peers", "send_bytes", "peer"),
                                                  ("recv_peers", "recv_bytes", "peer")):
                for item in row.get(field, []) or []:
                    edge = (rank, int(item[peer_field])) if direction == "send_bytes" else (int(item[peer_field]), rank)
                    edges[edge][direction] += int(item.get("bytes", 0))
        results.append({"comm_id": comm_id, "counter": group_id,
                        "status": status,
                        "root_cause": "NONE" if status == "P2P_DEVICE_EVIDENCE" else "P2P_COMPLETION_NEEDS_HOST_ASSOCIATION",
                        "confidence": "medium" if status == "P2P_DEVICE_EVIDENCE" else "low",
                        "first_error_rank": None,
                        "ranks": sorted(ranks), "records": len(rows),
                        "edge_evidence": {f"{a}->{b}": value for (a, b), value in sorted(edges.items())},
                        "evidence": {"association": "host op_count",
                                     "reason": "directional send/recv counters remain separate"}})
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
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for row in records:
        if row.get("phase") != "sample" or row.get("timestamp_ns") is None:
            continue
        node = int(row.get("node", -1))
        rank = int(row.get("rank", -1))
        channel = int(row.get("channel", -1))
        ts = int(row["timestamp_ns"])
        bucket = groups[(str(row.get("comm_id", "unknown")),
                         str(row.get("counter", "unknown")))][node][rank][channel]
        item = bucket.setdefault(ts, {"send_count": 0, "recv_count": 0})
        item["send_count"] += int(row.get("send_count", 0))
        item["recv_count"] += int(row.get("recv_count", 0))

    results = []
    for (comm_id, counter), nodes in sorted(groups.items()):
        stalled = []
        for node, ranks in nodes.items():
            totals = {}
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
                        if any(t1 >= start and t0 <= end and (send > 0 or recv > 0)
                               for t0, t1, send, recv in peer_intervals):
                            peer_progress.append(peer)
                    if peer_progress:
                        stalled.append({"rank": rank, "node": node,
                                        "channel": channel,
                                        "start_ns": start, "end_ns": end,
                                        "intervals": stagnant_intervals,
                                        "duration_ms": round((end - start) / 1_000_000, 3),
                                        "progressing_peers": sorted(peer_progress)})
                        break
        if stalled:
            slow_ranks = sorted({item["rank"] for item in stalled})
            result = {"comm_id": comm_id, "counter": counter,
                      "status": "COMM_STALLED", "slow_ranks": slow_ranks,
                      "stalled_intervals": stalled}
            result.update(_root_cause("COMM_STALLED", slow_ranks=slow_ranks))
            result["root_cause"] = "RANK_PROGRESS_STALLED"
            result["evidence"]["stalled_intervals"] = stalled
            results.append(result)
    return results
