import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccld_analyze import (analyze, analyze_completions,
                          analyze_completion_windows, analyze_progress_samples,
                          analyze_operation_semantics,
                          load_records)


def record(rank, *, op="AllReduce", byte_count=4096, ts=1_000_000_000,
           counter=7, nranks=2):
    return {"schema": 1, "phase": "enqueue", "comm_id": "abcd",
            "counter": counter, "trace_id": f"abcd:{counter}",
            "timestamp_ns": ts, "rank": rank, "nranks": nranks,
            "op": op, "count": byte_count // 4, "bytes": byte_count,
            "datatype": 7, "root": 0}


class CcldAnalyzeTest(unittest.TestCase):
    def test_p2p_edges_are_checked_reciprocally(self):
        rows = [record(0, op="Send"), record(1, op="Recv")]
        rows[0]["root"] = rows[1]["root"] = 1
        rows[1]["rank"] = 1
        result = analyze_operation_semantics(rows)[0]
        self.assertEqual(result["status"], "P2P_PEER_MISSING")

    def test_p2p_reciprocal_send_recv_is_clean(self):
        rows = [record(0, op="Send"), record(1, op="Recv")]
        rows[0]["root"] = 1
        rows[1]["root"] = 0
        result = analyze_operation_semantics(rows)[0]
        self.assertEqual(result["status"], "OK")

    def test_grouped_p2p_pairs_by_shared_op_count_not_per_call_counter(self):
        send = record(0, op="Send", counter=40, nranks=2)
        recv = record(1, op="Recv", counter=41, nranks=2)
        send.update(root=1, op_count=12, count=32, bytes=128)
        recv.update(root=0, op_count=12, count=32, bytes=128)
        results = analyze_operation_semantics([send, recv])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "OK")
        self.assertEqual(results[0]["counter"], "12")
        self.assertEqual(results[0]["grouping"], "op_count")

    def test_grouped_p2p_detects_size_mismatch_across_call_counters(self):
        send = record(0, op="Send", counter=50, nranks=2)
        recv = record(1, op="Recv", counter=51, nranks=2, byte_count=2048)
        send.update(root=1, op_count=13)
        recv.update(root=0, op_count=13)
        result = analyze_operation_semantics([send, recv])[0]
        self.assertEqual(result["status"], "P2P_TRANSFER_MISMATCH")
        self.assertEqual(result["evidence"]["edge_mismatches"][0]["fields"]["bytes"], [4096, 2048])

    def test_grouped_p2p_op_counts_remain_separate(self):
        rows = []
        for op_count in (14, 15):
            send = record(0, op="Send", counter=op_count * 2, nranks=2)
            recv = record(1, op="Recv", counter=op_count * 2 + 1, nranks=2)
            send.update(root=1, op_count=op_count)
            recv.update(root=0, op_count=op_count)
            rows.extend((send, recv))
        results = analyze_operation_semantics(rows)
        self.assertEqual(len(results), 2)
        self.assertEqual([row["counter"] for row in results], ["14", "15"])
        self.assertTrue(all(row["status"] == "OK" for row in results))

    def test_broadcast_root_role_is_checked(self):
        rows = [record(0, op="Broadcast"), record(1, op="Broadcast")]
        rows[0]["root"] = 0
        rows[1]["root"] = 1
        result = analyze_operation_semantics(rows)[0]
        self.assertEqual(result["status"], "BROADCAST_ROLE_MISMATCH")

    def test_alltoall_shape_is_checked(self):
        rows = [record(0, op="AllToAll"), record(1, op="AllToAll", byte_count=2048)]
        result = analyze_operation_semantics(rows)[0]
        self.assertEqual(result["status"], "ALLTOALL_SHAPE_MISMATCH")

    def test_consistent_two_rank_collective(self):
        result = analyze([record(0), record(1)], now_ns=1_000_000_001)[0]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["trace_id"], "abcd:7")

    def test_records_can_carry_stable_op_count(self):
        rows = [record(0), record(1)]
        rows[0]["op_count"] = rows[1]["op_count"] = 42
        result = analyze(rows, now_ns=1_000_000_001)[0]
        self.assertEqual(result["status"], "OK")
        self.assertEqual({row["op_count"] for row in rows}, {42})

    def test_detects_operation_mismatch(self):
        result = analyze([record(0), record(1, op="Broadcast")])[0]
        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(result["mismatches"]["op"], ["AllReduce", "Broadcast"])
        self.assertEqual(result["root_cause"], "PARAMETER_MISMATCH")
        self.assertEqual(result["confidence"], "high")

    def test_resource_exhaustion_is_observation_loss(self):
        dropped = record(0)
        dropped["diagnostic_status"] = "OBSERVATION_DROPPED"
        good = record(1)
        result = analyze([dropped, good])[0]
        self.assertEqual(result["status"], "OBSERVATION_DROPPED")
        self.assertEqual(result["root_cause"], "DIAGNOSTIC_RESOURCE_EXHAUSTED")
        self.assertEqual(result["dropped_ranks"], [0])
        completion = dict(dropped, phase="complete", send_count=0,
                          recv_count=0, algorithm=-1, protocol=-1)
        self.assertEqual(analyze_completions([dropped, completion]), [])

    def test_detects_count_and_byte_mismatch(self):
        result = analyze([record(0), record(1, byte_count=2048)])[0]
        self.assertEqual(result["status"], "MISMATCH")
        self.assertIn("count", result["mismatches"])
        self.assertIn("bytes", result["mismatches"])

    def test_detects_datatype_and_root_mismatch(self):
        peer = record(1)
        peer["datatype"] = 8
        peer["root"] = 1
        result = analyze([record(0), peer])[0]
        self.assertEqual(result["status"], "MISMATCH")
        self.assertIn("datatype", result["mismatches"])
        self.assertIn("root", result["mismatches"])

    def test_detects_missing_rank_after_timeout(self):
        result = analyze([record(0)], now_ns=4_000_000_000,
                         timeout_ms=1000)[0]
        self.assertEqual(result["status"], "MISSING_RANK")
        self.assertEqual(result["missing_ranks"], [1])
        self.assertEqual(result["first_error_rank"], 1)
        self.assertEqual(result["root_cause"], "RANK_DID_NOT_ENQUEUE")

    def test_reports_pending_before_timeout(self):
        result = analyze([record(0)], now_ns=1_100_000_000,
                         timeout_ms=1000)[0]
        self.assertEqual(result["status"], "PENDING")

    def test_detects_enqueue_skew(self):
        result = analyze([record(0), record(1, ts=8_000_000_000)],
                         now_ns=8_000_000_001, slow_skew_ms=2)[0]
        self.assertEqual(result["status"], "ENQUEUE_SKEW")

    def test_does_not_compare_enqueue_clock_across_nodes(self):
        first = record(0, ts=1_000_000_000, nranks=2)
        second = record(1, ts=8_000_000_000, nranks=2)
        first["nnodes"] = second["nnodes"] = 2
        result = analyze([first, second], now_ns=8_000_000_001,
                         slow_skew_ms=2)[0]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["nnodes"], 2)

    def test_jsonl_loader_ignores_partial_line_and_non_enqueue_events(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "rank-0.jsonl"
            with log.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(record(0)) + "\n")
                stream.write('{"phase":"complete","comm_id":"abcd","counter":7,"rank":0,"nranks":1,"send_count":8,"recv_count":8}\n')
                stream.write('{"phase":"sample","comm_id":"abcd","counter":7,"rank":0,"timestamp_ns":4}\n')
                stream.write('{"phase":"enqueue"')
            loaded = load_records([log])
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]["rank"], 0)

    def test_completion_counts_are_aggregated_per_rank(self):
        rows = []
        for rank in (0, 1):
            rows.extend({"phase": "complete", "comm_id": "abcd", "counter": 3,
                         "rank": rank, "nranks": 2, "channel": channel,
                         "send_count": 4, "recv_count": 4, "op_count": 3} for channel in (0, 1))
        result = analyze_completions(rows)[0]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["per_rank"][0]["send_count"], 8)
        self.assertEqual(result["op_counts"], ["3"])

    def test_completion_count_mismatch_is_detected(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 0, "nranks": 2, "send_count": 8, "recv_count": 8},
                {"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 1, "nranks": 2, "send_count": 4, "recv_count": 8}]
        self.assertEqual(analyze_completions(rows)[0]["status"], "COUNT_MISMATCH")

    def test_completion_uses_only_channel_counters(self):
        row = {"phase": "complete", "comm_id": "abcd", "counter": 3,
               "rank": 0, "nranks": 1, "channel": 0,
               "send_count": 8, "recv_count": 6,
               "peer_bytes": 999, "protocol_matrix": [1, 2, 3]}
        result = analyze_completions([row])[0]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["per_rank"][0]["send_count"], 8)
        self.assertNotIn("peer_bytes", result)

    def test_incomplete_completion_is_detected(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 0, "nranks": 2, "send_count": 8, "recv_count": 8,
                 "complete": True},
                {"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 1, "nranks": 2, "send_count": 8, "recv_count": 8,
                 "complete": False}]
        result = analyze_completions(rows)[0]
        self.assertEqual(result["status"], "MISSING_COMPLETION")
        self.assertEqual(result["incomplete_ranks"], [1])

    def test_enqueue_without_completion_waits_for_timeout(self):
        row = record(0, nranks=1)
        row["timestamp_wall_ns"] = 1_000_000_000
        pending = analyze_completions([row], now_ns=1_500_000_000,
                                      timeout_ms=1000)[0]
        self.assertEqual(pending["status"], "PENDING")
        expired = analyze_completions([row], now_ns=3_000_000_000,
                                      timeout_ms=1000)[0]
        self.assertEqual(expired["status"], "MISSING_COMPLETION")

    def test_completion_rate_outlier_is_detected(self):
        rows = []
        for rank, duration in ((0, 100), (1, 100), (2, 400), (3, 100)):
            rows.append({"phase": "complete", "comm_id": "abcd", "counter": 8,
                         "rank": rank, "nranks": 4, "send_count": 1000,
                         "recv_count": 1000, "start_timestamp": 0,
                         "end_timestamp": duration, "complete": True})
        result = analyze_completions(rows)[0]
        self.assertEqual(result["status"], "COMM_SLOW")
        self.assertEqual(result["slow_ranks"], [2])

    def test_completion_duration_jitter_does_not_override_directional_counters(self):
        rows = []
        for rank, duration, sends, receives in ((0, 100, 1000, 0),
                                                (1, 400, 0, 1000),
                                                (2, 105, 1000, 1000)):
            rows.append({"phase": "complete", "comm_id": "duration",
                         "counter": 1, "rank": rank, "nranks": 3,
                         "algorithm": 1, "protocol": 2,
                         "send_count": sends, "recv_count": receives,
                         "start_timestamp": 0, "end_timestamp": duration,
                         "complete": True})
        result = analyze_completions(rows)[0]
        self.assertEqual(result["status"], "TOPOLOGY_PENDING")

    def test_completion_slow_rate_waits_for_both_directions_on_all_ranks(self):
        rows = []
        for rank, send, recv in ((0, 0, 1000), (1, 1000, 1000),
                                 (2, 1000, 1000), (3, 1000, 1000)):
            rows.append({"phase": "complete", "comm_id": "bcast", "counter": 2,
                         "rank": rank, "nranks": 4, "op": "Broadcast",
                         "send_count": send, "recv_count": recv,
                         "start_timestamp": 0, "end_timestamp": 100})
        self.assertEqual(analyze_completions(rows)[0]["status"], "OK")

    def test_completion_keeps_peer_topology_edges(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 9,
                 "rank": 0, "nranks": 2, "send_count": 8, "recv_count": 8,
                 "send_rank": 1, "recv_rank": 1, "channel_base": 0},
                {"phase": "complete", "comm_id": "abcd", "counter": 9,
                 "rank": 1, "nranks": 2, "send_count": 8, "recv_count": 8,
                 "send_rank": 0, "recv_rank": 0, "channel_base": 0}]
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_edges"], [(0, 1, 1, 0), (1, 0, 0, 0)])

    def test_completion_aggregates_topology_groups_with_rates(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 10,
                 "rank": 0, "nranks": 2, "send_count": 8, "recv_count": 6,
                 "send_rank": 1, "recv_rank": 1, "channel_base": 2,
                 "start_timestamp": 0, "end_timestamp": 4},
                {"phase": "complete", "comm_id": "abcd", "counter": 10,
                 "rank": 1, "nranks": 2, "send_count": 6, "recv_count": 8,
                 "send_rank": 0, "recv_rank": 0, "channel_base": 2,
                 "start_timestamp": 0, "end_timestamp": 4}]
        result = analyze_completions(rows)[0]
        self.assertTrue(result["topology_ready"])
        self.assertEqual(result["topology_groups"][0]["send_count"], 8)
        self.assertEqual(result["topology_groups"][1]["send_rate"], 1.5)

    def test_ring_topology_is_symmetric(self):
        rows = []
        for rank, peer in ((0, 1), (1, 0)):
            rows.append({"phase": "complete", "comm_id": "abcd", "counter": 11,
                         "rank": rank, "nranks": 2, "algorithm": 1,
                         "channel": 0, "ring_prev": peer, "ring_next": peer,
                         "send_count": 8, "recv_count": 8,
                         "start_timestamp": 0, "end_timestamp": 4})
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "RING_SYMMETRIC")
        json.dumps(result)

    def test_ring_topology_detects_broken_neighbor_reciprocity(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 111,
                 "rank": rank, "nranks": 3, "algorithm": 1, "channel": 0,
                 "ring_prev": prev, "ring_next": nxt, "send_count": 8,
                 "recv_count": 8}
                for rank, prev, nxt in ((0, 2, 1), (1, 0, 2), (2, 0, 0))]
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "RING_ASYMMETRY")

    def test_ring_count_rate_outlier_is_slow_not_topology_fault(self):
        rows = []
        for rank, peer, count in ((0, 1, 8), (1, 0, 2)):
            rows.append({"phase": "complete", "comm_id": "abcd", "counter": 12,
                         "rank": rank, "nranks": 2, "algorithm": 1,
                         "send_count": count, "recv_count": count,
                         "channel": 0, "ring_prev": peer, "ring_next": peer,
                         "start_timestamp": 0, "end_timestamp": 4})
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "RING_SYMMETRIC")
        self.assertEqual(result["status"], "COMM_SLOW")

    def test_ring_topology_does_not_treat_duration_jitter_as_bad_links(self):
        rows = []
        for rank in range(4):
            rows.append({"phase": "complete", "comm_id": "abcd", "counter": 120,
                         "rank": rank, "nranks": 4, "algorithm": 1,
                         "channel": 0, "ring_prev": (rank - 1) % 4,
                         "ring_next": (rank + 1) % 4, "send_count": 1000,
                         "recv_count": 1000, "start_timestamp": 0,
                         "end_timestamp": (100, 100, 100, 1000)[rank]})
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "RING_SYMMETRIC")

    def test_ring_partial_rank_set_is_pending(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 121,
                 "rank": rank, "nranks": 4, "algorithm": 1, "channel": 2,
                 "ring_prev": (rank - 1) % 4, "ring_next": (rank + 1) % 4,
                 "send_count": 100, "recv_count": 100}
                for rank in (0, 1, 2)]
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "RING_TOPOLOGY_PENDING")
        self.assertEqual(result["topology_analysis"]["pending_channels"], [2])

    def test_tree_topology_requires_explicit_levels(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 13,
                 "rank": 0, "nranks": 2, "algorithm": 0, "send_count": 8,
                 "recv_count": 8, "tree_up": -1, "tree_down0": 1,
                 "tree_down1": -1, "tree_down2": -1,
                 "channel": 0}]
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "TREE_LEVEL_PENDING")

    def test_tree_topology_reconstructs_rank_levels_from_peers(self):
        rows = []
        topology = [(0, -1, 1, 2, -1, 0, 800),
                    (1, 0, 3, -1, -1, 1, 800),
                    (2, 0, -1, -1, -1, 1, 200),
                    (3, 1, -1, -1, -1, 2, 800)]
        for rank, up, d0, d1, d2, depth, count in topology:
            rows.append({"phase": "complete", "comm_id": "abcd", "counter": 14,
                         "rank": rank, "nranks": 4, "algorithm": 0,
                         "send_count": count, "recv_count": count,
                         "tree_up": up, "tree_down0": d0,
                         "tree_down1": d1, "tree_down2": d2, "tree_depth": 4,
                         "channel": 0, "start_timestamp": 0, "end_timestamp": 4})
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "TREE_SYMMETRIC")
        self.assertEqual(sorted({item["depth"] for item in result["topology_analysis"]["channel_depths"]}), [0, 1, 2])

    def test_tree_topology_detects_broken_parent_child_link(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 15,
                 "rank": rank, "nranks": 3, "algorithm": 0, "channel": 0,
                 "send_count": 8, "recv_count": 8, "tree_up": up,
                 "tree_down0": down, "tree_down1": -1, "tree_down2": -1,
                 "tree_depth": 3}
                for rank, up, down in ((0, -1, 2), (1, 0, -1), (2, 1, -1))]
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "TREE_ASYMMETRY")

    def test_tree_count_difference_waits_for_topology(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 0, "nranks": 2, "channel": 0, "algorithm": 0,
                 "protocol": 2, "send_count": 8, "recv_count": 8},
                {"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 1, "nranks": 2, "channel": 0, "algorithm": 0,
                 "protocol": 2, "send_count": 4, "recv_count": 4}]
        self.assertEqual(analyze_completions(rows)[0]["status"], "TOPOLOGY_PENDING")

    def test_ring_count_difference_waits_for_edge_topology(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 4,
                 "rank": 0, "nranks": 2, "channel": 0, "algorithm": 1,
                 "protocol": 0, "send_count": 8, "recv_count": 8},
                {"phase": "complete", "comm_id": "abcd", "counter": 4,
                 "rank": 1, "nranks": 2, "channel": 0, "algorithm": 1,
                 "protocol": 0, "send_count": 4, "recv_count": 4}]
        self.assertEqual(analyze_completions(rows)[0]["status"], "TOPOLOGY_PENDING")

    def test_multinode_unknown_algorithm_count_waits_for_topology(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 5,
                 "rank": 0, "nranks": 2, "nnodes": 2,
                 "send_count": 8, "recv_count": 8},
                {"phase": "complete", "comm_id": "abcd", "counter": 5,
                 "rank": 1, "nranks": 2, "nnodes": 2,
                 "send_count": 4, "recv_count": 4}]
        self.assertEqual(analyze_completions(rows)[0]["status"], "TOPOLOGY_PENDING")

    def test_completion_window_requires_persistent_slow_rank(self):
        rows = []
        for counter in (1, 2, 3):
            for rank, duration in ((0, 100), (1, 100), (2, 400)):
                rows.append({"phase": "complete", "comm_id": "abcd",
                             "counter": counter, "rank": rank, "nranks": 3,
                             "send_count": 1000, "recv_count": 1000,
                             "start_timestamp": 0, "end_timestamp": duration,
                             "complete": True})
        result = analyze_completion_windows(rows, window_size=3)[0]
        self.assertEqual(result["status"], "COMM_SLOW_WINDOW")
        self.assertEqual(result["slow_ranks"], [2])
        self.assertEqual(result["observed_windows"], 1)

    def test_completion_window_does_not_alert_one_noisy_iteration(self):
        rows = []
        for counter in (1, 2, 3):
            for rank in (0, 1, 2):
                duration = 400 if counter == 2 and rank == 2 else 100
                rows.append({"phase": "complete", "comm_id": "abcd",
                             "counter": counter, "rank": rank, "nranks": 3,
                             "send_count": 1000, "recv_count": 1000,
                             "start_timestamp": 0, "end_timestamp": duration,
                             "complete": True})
        self.assertEqual(analyze_completion_windows(rows, window_size=3), [])

    def test_progress_samples_detect_rank_stall_with_local_peer_progress(self):
        rows = []
        for ts, p0, p1 in ((0, 0, 0), (10, 0, 10), (20, 0, 20), (30, 0, 30)):
            rows.extend([
                {"phase": "sample", "comm_id": "abcd", "counter": 20,
                 "rank": 0, "node": 0, "timestamp_ns": ts,
                 "send_count": p0, "recv_count": p0},
                {"phase": "sample", "comm_id": "abcd", "counter": 20,
                 "rank": 1, "node": 0, "timestamp_ns": ts,
                 "send_count": p1, "recv_count": p1},
            ])
        result = analyze_progress_samples(rows, stagnant_intervals=3)[0]
        self.assertEqual(result["status"], "COMM_STALLED")
        self.assertEqual(result["slow_ranks"], [0])
        self.assertEqual(result["root_cause"], "RANK_PROGRESS_STALLED")
        self.assertEqual(result["stalled_intervals"][0]["progressing_peers"], [1])

    def test_progress_samples_identify_zero_progress_rank_when_peer_progress_precedes_sampling(self):
        rows = []
        for ts in (0, 10, 20, 30):
            rows.extend([
                {"phase": "sample", "comm_id": "early", "counter": 1,
                 "rank": 0, "node": 0, "local_rank": 0, "local_ranks": 2,
                 "timestamp_ns": ts, "channel": 0,
                 "send_count": 100, "recv_count": 100},
                {"phase": "sample", "comm_id": "early", "counter": 1,
                 "rank": 1, "node": 0, "local_rank": 1, "local_ranks": 2,
                 "timestamp_ns": ts, "channel": 0,
                 "send_count": 0, "recv_count": 0},
            ])
        result = analyze_progress_samples(rows, stagnant_intervals=3)[0]
        self.assertEqual(result["slow_ranks"], [1])
        self.assertEqual(result["root_cause"], "RANK_PROGRESS_STALLED")

    def test_progress_samples_detect_all_ranks_stalled_on_node(self):
        rows = []
        for ts in (0, 10, 20, 30):
            for rank in (0, 1):
                rows.append({"phase": "sample", "comm_id": "abcd", "counter": 22,
                             "rank": rank, "node": 0, "nranks": 2,
                             "timestamp_ns": ts, "channel": 0,
                             "send_count": 0, "recv_count": 0})
        result = analyze_progress_samples(rows, stagnant_intervals=3)[0]
        self.assertEqual(result["status"], "COMM_STALLED")
        self.assertEqual(result["scope"], "all_ranks")
        self.assertEqual(result["root_cause"], "ALL_RANKS_NO_PROGRESS")

    def test_multinode_progress_uses_local_rank_count(self):
        rows = []
        for ts in (0, 10, 20, 30):
            for rank, node, local_rank in ((0, 0, 0), (1, 0, 1),
                                           (2, 1, 0), (3, 1, 1)):
                rows.append({"phase": "sample", "comm_id": "multi", "counter": 1,
                             "rank": rank, "node": node, "local_rank": local_rank,
                             "local_ranks": 2, "nranks": 4, "timestamp_ns": ts,
                             "channel": 0, "send_count": 0, "recv_count": 0})
        result = analyze_progress_samples(rows, stagnant_intervals=3)[0]
        self.assertEqual(result["scope"], "all_ranks")
        self.assertEqual(result["slow_ranks"], [0, 1, 2, 3])

    def test_progress_samples_do_not_compare_different_nodes(self):
        rows = []
        for ts, p0, p1 in ((0, 0, 0), (10, 0, 10), (20, 0, 20), (30, 0, 30)):
            rows.extend([
                {"phase": "sample", "comm_id": "abcd", "counter": 21,
                 "rank": 0, "node": 0, "timestamp_ns": ts,
                 "send_count": p0, "recv_count": p0},
                {"phase": "sample", "comm_id": "abcd", "counter": 21,
                 "rank": 1, "node": 1, "timestamp_ns": ts,
                 "send_count": p1, "recv_count": p1},
            ])
        self.assertEqual(analyze_progress_samples(rows, stagnant_intervals=3), [])

    def test_progress_samples_ignore_counter_reset_intervals(self):
        rows = []
        for i, value in enumerate((100, 110, 0, 0, 0)):
            rows.append({"phase": "sample", "comm_id": "reset", "counter": 9,
                         "timestamp_ns": (i + 1) * 10_000_000,
                         "rank": 0, "node": 0, "channel": 0,
                         "send_count": value, "recv_count": value,
                         "counter_reset": i == 2})
        result = analyze_progress_samples(rows, stagnant_intervals=3)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
