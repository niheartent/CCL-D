import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccld_analyze import (analyze, analyze_completions,
                          analyze_completion_windows, load_records)


def record(rank, *, op="AllReduce", byte_count=4096, ts=1_000_000_000,
           counter=7, nranks=2):
    return {"schema": 1, "phase": "enqueue", "comm_id": "abcd",
            "counter": counter, "trace_id": f"abcd:{counter}",
            "timestamp_ns": ts, "rank": rank, "nranks": nranks,
            "op": op, "count": byte_count // 4, "bytes": byte_count,
            "datatype": 7, "root": 0}


class CcldAnalyzeTest(unittest.TestCase):
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
                stream.write('{"phase":"enqueue"')
            loaded = load_records([log])
        self.assertEqual(len(loaded), 2)
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
        self.assertAlmostEqual(result["per_rank"][2]["send_rate"], 2.5)

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

    def test_ring_topology_detects_asymmetric_edge_rate(self):
        rows = []
        for rank, peer, count in ((0, 1, 8), (1, 0, 2)):
            rows.append({"phase": "complete", "comm_id": "abcd", "counter": 12,
                         "rank": rank, "nranks": 2, "algorithm": 1,
                         "send_count": count, "recv_count": count,
                         "channel": 0, "ring_prev": peer, "ring_next": peer,
                         "start_timestamp": 0, "end_timestamp": 4})
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "RING_ASYMMETRY")

    def test_tree_topology_requires_explicit_levels(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 13,
                 "rank": 0, "nranks": 2, "algorithm": 0, "send_count": 8,
                 "recv_count": 8, "tree_up": -1, "tree_down0": 1,
                 "tree_down1": -1, "tree_down2": -1,
                 "channel": 0}]
        result = analyze_completions(rows)[0]
        self.assertEqual(result["topology_analysis"]["status"], "TREE_LEVEL_PENDING")

    def test_tree_topology_compares_rates_by_level(self):
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
                         "tree_down1": d1, "tree_down2": d2, "tree_depth": depth,
                         "channel": 0, "start_timestamp": 0, "end_timestamp": 4})
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


if __name__ == "__main__":
    unittest.main()
