#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SIM="$ROOT/custom-rccl-dtk-26.04-ccl-d/tools/tests/simulate_ccld_device_fault.py"
MON="$ROOT/custom-rccl-dtk-26.04-ccl-d/tools/ccld_monitor.py"
OUT=${CCLD_SIM_DIR:-$ROOT/ccld-artifacts/fault-simulation-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
run_case() {
  local name=$1 fault=$2 expected=$3
  local log="$OUT/$name.jsonl"
  python3 "$SIM" "$log" --ranks 8 --channels 2 --fault "$fault"
  set +e
  python3 "$MON" --once --timeout-ms 0 --fail-on-anomaly "$log" >"$OUT/$name.monitor.jsonl"
  local status=$?
  set -e
  if [[ $status -ne 2 ]]; then echo "$name: expected monitor exit 2, got $status" >&2; return 1; fi
  if ! grep -q "\"status\": \"$expected\"" "$OUT/$name.monitor.jsonl"; then cat "$OUT/$name.monitor.jsonl" >&2; return 1; fi
  echo "$name: PASS ($expected)"
}
run_case stall stall COUNT_MISMATCH
run_case slow slow COUNT_MISMATCH
run_case missing missing MISSING_COMPLETION
echo "fault_simulation_matrix=PASS dir=$OUT"
