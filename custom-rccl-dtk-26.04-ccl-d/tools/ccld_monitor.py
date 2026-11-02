#!/usr/bin/env python3
"""Monitor appended RCCL CCL-D JSONL logs on one host."""
import argparse
import glob
import json
import pathlib
import time

from ccld_analyze import analyze, analyze_completions, load_records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="JSONL paths or shell-expanded globs")
    parser.add_argument("--once", action="store_true", help="analyze once and exit")
    parser.add_argument("--poll-ms", type=int, default=250)
    parser.add_argument("--timeout-ms", type=int, default=2000,
                        help="quiet time before reporting missing ranks")
    parser.add_argument("--slow-skew-ms", type=float, default=2.0,
                        help="enqueue-time spread threshold within one host")
    parser.add_argument("--all", action="store_true", help="print OK and PENDING groups too")
    parser.add_argument("--fail-on-anomaly", action="store_true",
                        help="return 2 when a mismatch, missing rank, or slow skew is found")
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
        results += analyze_completions(records)
        for result in results:
            status = result["status"]
            anomaly = status in ("MISMATCH", "MISSING_RANK", "ENQUEUE_SKEW",
                                 "COUNT_MISMATCH", "MISSING_COMPLETION")
            if anomaly:
                had_anomaly = True
            if not (args.all or anomaly):
                continue
            signature = (result["comm_id"], result["counter"], status,
                         tuple(result.get("missing_ranks", [])),
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
