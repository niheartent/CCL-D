#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
VALIDATE="$ROOT/custom-rccl-dtk-26.04-ccl-d/tools/tests/run_multinode_ccld_validation.sh"
BASE=${CCLD_MATRIX_DIR:-$ROOT/ccld-artifacts/fault-matrix-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$BASE"
run_case() {
  local name=$1 fault=$2 expected=$3
  local dir="$BASE/$name"
  echo "case=$name fault=$fault expected=$expected"
  set +e
  CCLD_OUT_DIR="$dir" CCLD_ITERS=20 CCLD_COUNT=4096 CCLD_FAULT="$fault" CCLD_FAULT_RANK=7 CCLD_FAULT_COUNTER=10 "$VALIDATE" >"$dir.console" 2>&1
  local status=$?
  set -e
  if [[ "$expected" == OK && $status -ne 0 ]]; then cat "$dir.console" >&2; return 1; fi
  if [[ "$expected" != OK && $status -ne 2 ]]; then cat "$dir.console" >&2; return 1; fi
  if [[ "$expected" != OK ]] && ! grep -q "\"status\": \"$expected\"" "$dir/monitor.jsonl"; then cat "$dir/monitor.jsonl" >&2; return 1; fi
  echo "$name: PASS"
}
run_case normal "" OK
run_case op_mismatch op_mismatch MISMATCH
run_case size_mismatch size_mismatch MISMATCH
run_case missing_record skip_record MISSING_RANK
run_case missing_rank missing_rank MISSING_RANK
echo "fault_matrix=PASS dir=$BASE"
