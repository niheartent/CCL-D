#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/rccl-UnitTests" >&2
  exit 64
fi
unit_tests=$(realpath "$1")
libdir=${RCCL_CCLD_LIBDIR:?set RCCL_CCLD_LIBDIR to the CCL-D install lib directory}
monitor="$(cd "$(dirname "$0")/.." && pwd)/ccld_monitor.py"
workdir=$(mktemp -d "ccld-runtime-test.XXXXXX")
trap 'rm -rf "$workdir"' EXIT

if [[ -z ${HIP_PATH:-} && -r /opt/dtk-26.04/env.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/dtk-26.04/env.sh
fi
export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RCCL_CCLD_ENABLE=1
# Avoid single-rank gtest subcases: RCCL opCount is reused at nranks=1.
export UT_MIN_GPUS=2
export RCCL_CCLD_LOG="$workdir/rank-{rank}.jsonl"

"$unit_tests" --gtest_filter='AllReduce.OutOfPlace'
shopt -s nullglob
log_files=("$workdir"/rank-*.jsonl)
if [[ ${#log_files[@]} -eq 0 ]]; then
  echo "no rank log files were created" >&2
  exit 1
fi
python3 - "${log_files[@]}" <<'PY'
import json, sys
seen = set()
completions = []
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if row.get("phase") == "complete":
            completions.append(row)
        key = (str(row["comm_id"]), int(row["rank"]), int(row["counter"]))
        if row.get("phase") == "enqueue" and key in seen:
            raise SystemExit(f"duplicate per-rank diagnostic counter: {key}")
        if row.get("phase") == "enqueue":
            seen.add(key)
if not seen:
    raise SystemExit("no enqueue diagnostic records")
if not completions:
    raise SystemExit("no complete diagnostic records")
if any("send_count" not in row or "recv_count" not in row for row in completions):
    raise SystemExit("completion record missing send/recv counters")
PY
python3 "$monitor" --once --all --fail-on-anomaly "${log_files[@]}" \
  --slow-skew-ms 60000 > "$workdir/normal-monitor.jsonl"
grep -q '"status": "OK"' "$workdir/normal-monitor.jsonl"
grep -Eq '"status": "(OK|TOPOLOGY_PENDING|RING_ASYMMETRY|TREE_ASYMMETRY)"' "$workdir/normal-monitor.jsonl"

echo "runtime smoke=PASS"
