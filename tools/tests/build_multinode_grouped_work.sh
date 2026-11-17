#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source /opt/dtk-26.04/env.sh
OUT=${CCLD_GROUPED_EXE:-$ROOT/rccl-artifacts/multinode-test/multinode_grouped_work}
PREFIX=${RCCL_PREFIX:-$ROOT/rccl-artifacts/ccld-frame-gfx936/install}
mkdir -p "$(dirname "$OUT")"
/opt/mpi/bin/mpicxx -D__HIP_PLATFORM_AMD__ "$ROOT/tools/tests/multinode_grouped_work.cpp" \
  -I"$PREFIX/include" -L"$PREFIX/lib" -L/opt/dtk-26.04/hip/lib \
  -Wl,-rpath,"$PREFIX/lib" -Wl,-rpath,/opt/dtk-26.04/hip/lib \
  -lrccl -lamdhip64 -o "$OUT"
printf 'built %s\n' "$OUT"
