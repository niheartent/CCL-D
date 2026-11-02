import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccld_analyze import analyze, analyze_completions, load_records


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
                         "send_count": 4, "recv_count": 4} for channel in (0, 1))
        result = analyze_completions(rows)[0]
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["per_rank"][0]["send_count"], 8)

    def test_completion_count_mismatch_is_detected(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 0, "nranks": 2, "send_count": 8, "recv_count": 8},
                {"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 1, "nranks": 2, "send_count": 4, "recv_count": 8}]
        self.assertEqual(analyze_completions(rows)[0]["status"], "COUNT_MISMATCH")

    def test_tree_count_difference_waits_for_topology(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 0, "nranks": 2, "channel": 0, "algorithm": 1,
                 "protocol": 2, "send_count": 8, "recv_count": 8},
                {"phase": "complete", "comm_id": "abcd", "counter": 3,
                 "rank": 1, "nranks": 2, "channel": 0, "algorithm": 1,
                 "protocol": 2, "send_count": 4, "recv_count": 4}]
        self.assertEqual(analyze_completions(rows)[0]["status"], "TOPOLOGY_PENDING")

    def test_ring_count_difference_waits_for_edge_topology(self):
        rows = [{"phase": "complete", "comm_id": "abcd", "counter": 4,
                 "rank": 0, "nranks": 2, "channel": 0, "algorithm": 0,
                 "protocol": 0, "send_count": 8, "recv_count": 8},
                {"phase": "complete", "comm_id": "abcd", "counter": 4,
                 "rank": 1, "nranks": 2, "channel": 0, "algorithm": 0,
                 "protocol": 0, "send_count": 4, "recv_count": 4}]
        self.assertEqual(analyze_completions(rows)[0]["status"], "TOPOLOGY_PENDING")


if __name__ == "__main__":
    unittest.main()
