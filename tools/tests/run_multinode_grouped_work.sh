#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXE=${CCLD_GROUPED_EXE:-$ROOT/rccl-artifacts/multinode-test/multinode_grouped_work}
OUT=${CCLD_GROUPED_OUT_DIR:-$ROOT/rccl-artifacts/grouped-work-$(date -u +%Y%m%dT%H%M%SZ)}
PREFIX=${RCCL_PREFIX:-$ROOT/rccl-artifacts/ccld-frame-gfx936/install}
UCX_PREFIX=${UCX_PREFIX:-/opt/ucx-1.16.0}
HOSTS=${CCLD_HOSTS:-HPDPS-6:4,HPDPS-7:4}
NP=${CCLD_NP:-8}
WORKS=${CCLD_GROUP_WORKS:-256}
ITERS=${CCLD_ITERS:-1}
COUNT=${CCLD_COUNT:-4096}
DIAG_ENABLE=${CCLD_DIAG_ENABLE:-1}
test -x "$EXE"
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$PREFIX/lib:$UCX_PREFIX/lib:/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/mpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RCCL_CCLD_ENABLE=$DIAG_ENABLE RCCL_CCLD_SLOTS=${RCCL_CCLD_SLOTS:-512} RCCL_CCLD_SAMPLE_US=${RCCL_CCLD_SAMPLE_US:-20000}
export RCCL_CCLD_LOG="$OUT/rank-{rank}.jsonl"
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  /opt/mpi/bin/mpirun --mca plm_rsh_args "-p 2234" --host "$HOSTS" \
  --map-by ppr:4:node --bind-to none -np "$NP" \
  -x LD_LIBRARY_PATH -x RCCL_CCLD_ENABLE -x RCCL_CCLD_SLOTS -x RCCL_CCLD_SAMPLE_US -x RCCL_CCLD_LOG \
  "$EXE" "$WORKS" "$ITERS" "$COUNT" | tee "$OUT/run.log"
if [[ "$DIAG_ENABLE" == "1" ]]; then
  python3 "$ROOT/tools/ccld_monitor.py" --once --all --fail-on-anomaly --slow-rate-ratio 0 "$OUT"/rank-*.jsonl | tee "$OUT/monitor.jsonl"
  python3 - "$WORKS" "$ITERS" "$OUT"/rank-*.jsonl <<'PY'
import json, sys, collections
works, iterations = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(line) for path in sys.argv[3:] for line in open(path, encoding='utf-8')]
complete = [row for row in rows if row.get('phase') == 'complete']
resource = [row for row in rows if row.get('phase') == 'resource']
if not complete: raise SystemExit('no completion records')
allocated = [row for row in resource if row.get('resource_status') == 'allocated']
thread_done = [row for row in resource if row.get('resource_status') == 'thread_complete']
ranks = set(row['rank'] for row in rows if 'rank' in row)
if len(allocated) != len(ranks) or len(thread_done) != len(ranks):
    raise SystemExit('expected one frame allocation record per rank')
if any(row.get('thread_cpu_ns') is None or row.get('thread_wall_ns', 0) <= 0 or
       row.get('diagnostic_sample_cpu_ns') is None or row.get('sample_interval_ns', 0) <= 0
       for row in thread_done):
    raise SystemExit('missing diagnostic thread CPU measurements')
if any(row.get('frame_block_bytes') != 2144 or row.get('frame_pinned_bytes') != 2144 * row.get('slots', 0) for row in allocated):
    raise SystemExit('pinned frame byte accounting does not match frame layout')
if any(row.get('send_count', 0) < 0 or row.get('recv_count', 0) < 0 for row in complete): raise SystemExit('invalid counters')
if any(row.get('diagnostic_status') == 'OBSERVATION_DROPPED' for row in rows): raise SystemExit('diagnostic observations were dropped')
per_rank = collections.defaultdict(set)
for row in complete: per_rank[row['rank']].add((row.get('comm_id'), row.get('counter')))
expected = (works + 2) * iterations
if any(len(per_rank[rank]) != expected for rank in range(len(per_rank))):
    raise SystemExit(f'expected {expected} unique operation counters per rank, got {[len(per_rank[r]) for r in sorted(per_rank)]}')
print('grouped_complete_records=', len(complete), 'resource_records=', len(resource), 'unique_operation_counters_per_rank=', expected)
PY
else
  echo 'diagnostics disabled; communication and VmPin probe results are in run.log'
fi
