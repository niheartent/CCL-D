#!/usr/bin/env python3
"""Generate deterministic CCL-D completion records for container regression tests.

This does not touch a GPU. It models the diagnostic buffer after a kernel has
completed (or stopped making progress), allowing monitor decisions to be tested
in CI and in the Docker image.
"""
import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--fault", choices=("none", "stall", "slow", "missing"), default="none")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for rank in range(args.ranks):
            if args.fault == "missing" and rank == args.ranks - 1:
                continue
            total = 1024 if args.fault != "slow" or rank != 2 else 256
            if args.fault == "stall" and rank == 2:
                total = 0
            for channel in range(args.channels):
                row = {"schema": 1, "phase": "complete", "comm_id": "simulated",
                       "counter": 0, "trace_id": "simulated:0", "rank": rank,
                       "nranks": args.ranks, "node": 0, "nnodes": 1,
                       "channel": channel, "send_count": total,
                       "recv_count": total, "complete": args.fault != "stall" or rank != 2}
                stream.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
