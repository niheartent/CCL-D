#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARTIFACTS="$ROOT/rccl-artifacts/ccld-frame-gfx936"
LOG="$ARTIFACTS/build-independent-thread.log"
STATUS="$ARTIFACTS/build-independent-thread.status"
printf 'running\n' >"$STATUS"
cmake --build "$ARTIFACTS/build" -j8 >"$LOG" 2>&1
rc=$?
if [[ $rc -eq 0 ]]; then
  cmake --install "$ARTIFACTS/build" >>"$LOG" 2>&1
  rc=$?
fi
printf 'exit_code=%s\n' "$rc" >"$STATUS"
exit "$rc"
