#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXE=${CCLD_FAULT_EXE:-$ROOT/rccl-artifacts/multinode-test/multinode_allreduce_smoke}
OUT=${CCLD_FAULT_OUT_DIR:-$ROOT/rccl-artifacts/ccld-frame-gfx936/multinode-faults-$(date -u +%Y%m%dT%H%M%SZ)}
UCX_PREFIX=${UCX_PREFIX:-/opt/ucx-1.16.0}
HOSTS=${CCLD_HOSTS:-HPDPS-6:4,HPDPS-7:4}
NP=${CCLD_NP:-8}

source /opt/dtk-26.04/env.sh
unset UCX_HOME
if [[ -e "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -print -quit)" ]]; then
  echo "output directory is not empty: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$ROOT/rccl-artifacts/ccld-frame-gfx936/install/lib:$UCX_PREFIX/lib:/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/mpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RCCL_CCLD_ENABLE=1 RCCL_CCLD_SLOTS=512 RCCL_CCLD_SAMPLE_US=1000
export CCLD_ITERS=1 CCLD_COUNT=4096 CCLD_FAULT_RANK=2 CCLD_FAULT_DELAY_MS=2000
export CCLD_FAULT_STALL_MS=${CCLD_FAULT_STALL_MS:-30000}
export NCCL_PROTO=Simple NCCL_ALGO=Ring

run_case() {
  local mode=$1 timeout_s=$2 rc=0
  local dir="$OUT/$mode"
  mkdir -p "$dir"
  export CCLD_FAULT_MODE=$mode RCCL_CCLD_LOG="$dir/rank-{rank}.jsonl"
  python3 "$ROOT/tools/ccld_monitor.py" --poll-ms 50 --timeout-ms 0 --all \
    "$dir"/rank-*.jsonl > "$dir/monitor-live.jsonl" &
  local monitor_pid=$!
  sleep 1
  set +e
  OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    timeout --signal=TERM "${timeout_s}s" \
    /opt/mpi/bin/mpirun --mca plm_rsh_args "-p 2234" --host "$HOSTS" \
    --map-by ppr:4:node --bind-to none -np "$NP" \
    -x LD_LIBRARY_PATH -x RCCL_CCLD_ENABLE -x RCCL_CCLD_SLOTS \
    -x RCCL_CCLD_SAMPLE_US -x RCCL_CCLD_LOG -x CCLD_ITERS \
    -x CCLD_COUNT -x CCLD_FAULT_MODE -x CCLD_FAULT_RANK \
    -x CCLD_FAULT_DELAY_MS -x CCLD_FAULT_STALL_MS \
    -x NCCL_PROTO -x NCCL_ALGO "$EXE" > "$dir/run.log" 2>&1
  rc=$?
  set -e
  sleep 0.2
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  python3 - "$mode" "$dir" "$rc" <<'PY'
import json, pathlib, sys
mode, directory, rc = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
records = [json.loads(line) for line in (directory / "monitor-live.jsonl").read_text().splitlines() if line]
if mode == "slow":
    hits = [row for row in records if row.get("status") == "COMM_STALLED" and
            row.get("root_cause") == "RANK_PROGRESS_STALLED" and
            row.get("slow_ranks") == [2]]
    if rc != 0 or not hits:
        raise SystemExit(f"slow injection failed: run_rc={rc}, detections={len(hits)}")
elif mode == "missing":
    if rc == 0 or not any(row.get("status") == "MISSING_RANK" and
                          2 in row.get("missing_ranks", []) for row in records):
        raise SystemExit(f"missing-rank injection failed: run_rc={rc}")
elif mode == "stall":
    hits = [row for row in records if row.get("status") == "COMM_STALLED" and
            row.get("slow_ranks") == [2]]
    if rc != 124 or not hits:
        raise SystemExit(f"stall injection failed: run_rc={rc}, detections={len(hits)}")
print(f"{mode}=PASS run_rc={rc} monitor_records={len(records)}")
PY
}

run_case slow 10
run_case missing 10
run_case stall 8
echo "multinode_fault_validation=PASS output=$OUT"
