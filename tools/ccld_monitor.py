#!/usr/bin/env python3
"""Monitor appended RCCL CCL-D JSONL logs on one host."""
import argparse
import glob
import json
import pathlib
import time

from ccld_analyze import (analyze, analyze_completions,
                          analyze_completion_windows, analyze_progress_samples,
                          analyze_operation_semantics,
                          load_records)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="JSONL paths or shell-expanded globs")
    parser.add_argument("--once", action="store_true", help="analyze once and exit")
    parser.add_argument("--poll-ms", type=int, default=250)
    parser.add_argument("--timeout-ms", type=int, default=2000,
                        help="quiet time before reporting missing ranks")
    parser.add_argument("--slow-skew-ms", type=float, default=2.0,
                        help="enqueue-time spread threshold within one host")
    parser.add_argument("--slow-rate-ratio", type=float, default=0.5,
                        help="flag a rank whose device send rate is below this fraction of the median")
    parser.add_argument("--slow-window", type=int, default=0,
                        help="also require a slow rank to persist across this many collectives (0 disables)")
    parser.add_argument("--stagnant-intervals", type=int, default=3,
                        help="sample intervals with zero progress before reporting COMM_STALLED")
    parser.add_argument("--all", action="store_true", help="print OK and PENDING groups too")
    parser.add_argument("--fail-on-anomaly", action="store_true",
                        help="return 2 when a mismatch, missing rank, or slow skew is found")
    parser.add_argument("--strict-topology", action="store_true",
                        help="treat Ring/Tree topology asymmetry as a failing anomaly")
    args = parser.parse_args()
    seen = set()
    had_anomaly = False
    while True:
        paths = []
        for item in args.logs:
            matches = glob.glob(item)
            paths.extend(matches if matches else [item])
        records = load_records(paths)
        results = analyze(load_records(paths), timeout_ms=args.timeout_ms,
                          slow_skew_ms=args.slow_skew_ms)
        results += analyze_completions(records, slow_rate_ratio=args.slow_rate_ratio)
        results += analyze_operation_semantics(records)
        if args.slow_window:
            results += analyze_completion_windows(
                records, window_size=args.slow_window,
                slow_rate_ratio=args.slow_rate_ratio,
                min_slow_windows=max(2, args.slow_window // 2))
        results += analyze_progress_samples(
            records, stagnant_intervals=args.stagnant_intervals)
        for result in results:
            status = result["status"]
            anomaly = status in ("MISMATCH", "MISSING_RANK", "ENQUEUE_SKEW",
                                 "COUNT_MISMATCH", "MISSING_COMPLETION", "COMM_SLOW",
                                 "COMM_SLOW_WINDOW", "COMM_STALLED",
                                 "COMPUTATION_SLOW", "MIXED_SLOW",
                                 "P2P_PEER_MISSING", "P2P_TRANSFER_MISMATCH", "BROADCAST_ROLE_MISMATCH",
                                 "ALLTOALL_RANK_MISSING", "ALLTOALL_SHAPE_MISMATCH",
                                 "COLLECTIVE_SHAPE_MISMATCH", "DIAGNOSTIC_BATCH_DROPPED",
                                 "OBSERVATION_DROPPED")
            if args.strict_topology and status in ("RING_ASYMMETRY", "TREE_ASYMMETRY"):
                anomaly = True
            if anomaly:
                had_anomaly = True
            if not (args.all or anomaly):
                continue
            signature = (result["comm_id"], result["counter"], status,
                         result.get("operation"), result.get("root_cause"),
                         tuple(result.get("missing_ranks", [])),
                         tuple(result.get("incomplete_ranks", [])),
                         tuple(result.get("slow_ranks", [])),
                         json.dumps(result.get("stalled_intervals", []), sort_keys=True),
                         json.dumps(result.get("mismatches", {}), sort_keys=True),
                         tuple(result.get("send_counts", [])),
                         tuple(result.get("recv_counts", [])))
            if signature not in seen:
                print(json.dumps(result, sort_keys=True), flush=True)
                seen.add(signature)
        if args.once:
            return 2 if args.fail_on_anomaly and had_anomaly else 0
        time.sleep(max(args.poll_ms, 25) / 1000.0)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
