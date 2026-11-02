#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
EXE=${CCLD_SMOKE_EXE:-$ROOT/ccld-artifacts/multinode-test/multinode_allreduce_smoke}
OUT=${CCLD_OUT_DIR:-$ROOT/ccld-artifacts/validation-$(date -u +%Y%m%dT%H%M%SZ)}
UCX_PREFIX=${UCX_PREFIX:-/opt/ucx-1.16.0}
RCCL_PREFIX=${RCCL_PREFIX:-$ROOT/ccld-artifacts/dtk-26.04-gfx936-try1/install}
HOSTS=${CCLD_HOSTS:-HPDPS-6:4,HPDPS-7:4}
NP=${CCLD_NP:-8}
mkdir -p "$OUT"
test -x "$EXE"
export LD_LIBRARY_PATH="$RCCL_PREFIX/lib:$UCX_PREFIX/lib:/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/mpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset UCX_HOME
export RCCL_CCLD_ENABLE=1 RCCL_CCLD_LOG="$OUT/rank-{rank}.jsonl"
echo "validation_dir=$OUT"
echo "ucx=$($UCX_PREFIX/bin/ucx_info -v | sed -n '1p')"
echo "hosts=$HOSTS np=$NP"
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --mca plm_rsh_args "-p 2234" --host "$HOSTS" --map-by ppr:4:node --bind-to none -np "$NP" -x LD_LIBRARY_PATH -x RCCL_CCLD_ENABLE -x RCCL_CCLD_LOG "$EXE" 2>&1 | tee "$OUT/run.log"
set +e
python3 "$ROOT/custom-rccl-dtk-26.04-ccl-d/tools/ccld_monitor.py" --once --timeout-ms 0 --fail-on-anomaly "$OUT"/rank-*.jsonl | tee "$OUT/monitor.jsonl"
status=${PIPESTATUS[0]}
set -e
echo "monitor_exit=$status"
exit "$status"
