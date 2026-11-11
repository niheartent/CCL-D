#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXE=${CCLD_SLOT_REGRESSION_EXE:-$ROOT/ccld-artifacts/multinode-test/multinode_collective_slot_regression}
OUT=${CCLD_OUT_DIR:-$ROOT/ccld-artifacts/slot-regression-$(date -u +%Y%m%dT%H%M%SZ)}
RCCL_PREFIX=${RCCL_PREFIX:-$ROOT/ccld-artifacts/sample-probe-install}
UCX_PREFIX=${UCX_PREFIX:-/opt/ucx-1.16.0}
ITERS=${CCLD_ITERS:-20}
COUNT=${CCLD_COUNT:-4096}
mkdir -p "$OUT"
test -x "$EXE"
export LD_LIBRARY_PATH="$RCCL_PREFIX/lib:$UCX_PREFIX/lib:/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/mpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RCCL_CCLD_ENABLE=1 RCCL_CCLD_LOG="$OUT/rank-{rank}.jsonl"
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --mca plm_rsh_args "-p 2234" --host "${CCLD_HOSTS:-HPDPS-6:4,HPDPS-7:4}" --map-by ppr:4:node --bind-to none -np "${CCLD_NP:-8}" -x LD_LIBRARY_PATH -x RCCL_CCLD_ENABLE -x RCCL_CCLD_LOG "$EXE" "$ITERS" "$COUNT" | tee "$OUT/run.log"
python3 "$ROOT/tools/ccld_monitor.py" --once --all --slow-rate-ratio "${CCLD_SLOW_RATE_RATIO:-0}" "$OUT"/rank-*.jsonl | tee "$OUT/monitor.jsonl"
python3 - "$OUT"/rank-*.jsonl <<'PY'
import json, sys
rows = [json.loads(line) for path in sys.argv[1:] for line in open(path, encoding='utf-8')]
completions = [row for row in rows if row.get('phase') == 'complete']
if not completions:
    raise SystemExit('no completion records')
if any(row.get('peer_overflow') for row in completions):
    raise SystemExit('peer overflow detected')
if any(row.get('status') == 'PEER_DATA_INCONSISTENT' for row in rows):
    raise SystemExit('peer/protocol inconsistency detected')
print('slot_regression_records=', len(completions))
PY
echo "monitor_exit=0 (P2P pairing semantics are reported for follow-up analyzer work)"
