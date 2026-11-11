#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source /opt/dtk-26.04/env.sh
OUT=${CCLD_SLOT_REGRESSION_EXE:-$ROOT/ccld-artifacts/multinode-test/multinode_collective_slot_regression}
mkdir -p "$(dirname "$OUT")"
/opt/mpi/bin/mpicxx -D__HIP_PLATFORM_AMD__ \
  "$ROOT/tools/tests/multinode_collective_slot_regression.cpp" \
  -I"${RCCL_PREFIX:-$ROOT/ccld-artifacts/sample-probe-install}/include" \
  -L"${RCCL_PREFIX:-$ROOT/ccld-artifacts/sample-probe-install}/lib" \
  -L/opt/dtk-26.04/hip/lib \
  -Wl,-rpath,"${RCCL_PREFIX:-$ROOT/ccld-artifacts/sample-probe-install}/lib" \
  -Wl,-rpath,/opt/dtk-26.04/hip/lib \
  -lrccl -lamdhip64 -o "$OUT"
printf 'built %s\n' "$OUT"
