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
export RCCL_CCLD_LOG="$workdir/rank-{rank}.jsonl"
unset RCCL_CCLD_FAULT RCCL_CCLD_FAULT_RANK RCCL_CCLD_FAULT_COUNTER

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
grep -Eq '"status": "(OK|TOPOLOGY_PENDING)"' "$workdir/normal-monitor.jsonl"

run_injected_fault() {
  local fault=$1 expected=$2 monitor_status
  local -a injected_logs
  rm -f "$workdir"/rank-*.jsonl "$workdir/injected-monitor.jsonl"
  export RCCL_CCLD_FAULT="$fault"
  export RCCL_CCLD_FAULT_RANK=0
  export RCCL_CCLD_FAULT_COUNTER=0
  # The test must still succeed: injection changes diagnostic output only.
  "$unit_tests" --gtest_filter='AllReduce.OutOfPlace'
  injected_logs=("$workdir"/rank-*.jsonl)
  if [[ ${#injected_logs[@]} -eq 0 ]]; then
    echo "$fault: no rank log files were created" >&2
    exit 1
  fi
  set +e
  python3 "$monitor" --once --timeout-ms 0 --fail-on-anomaly \
    "${injected_logs[@]}" > "$workdir/injected-monitor.jsonl"
  monitor_status=$?
  set -e
  if [[ $monitor_status -ne 2 ]]; then
    cat "$workdir/injected-monitor.jsonl" >&2
    echo "$fault: expected monitor exit code 2, got $monitor_status" >&2
    exit 1
  fi
  if ! grep -q "\"status\": \"$expected\"" "$workdir/injected-monitor.jsonl"; then
    cat "$workdir/injected-monitor.jsonl" >&2
    echo "$fault: expected $expected diagnostic" >&2
    exit 1
  fi
  cat "$workdir/injected-monitor.jsonl"
}

# Mismatch injections need at least two ranks. The missing-rank injection also
# works with a one-rank test by altering nranks in diagnostic metadata only.
if python3 - "${log_files[@]}" <<'PY'
import json, sys
records = [json.loads(line) for path in sys.argv[1:] for line in open(path)]
raise SystemExit(0 if any(int(row.get("nranks", 1)) > 1 for row in records) else 1)
PY
then
  run_injected_fault op_mismatch MISMATCH
  run_injected_fault size_mismatch MISMATCH
else
  echo "SKIP op/size runtime injection: AllReduce test exposed only one rank"
fi
run_injected_fault missing_rank MISSING_RANK
