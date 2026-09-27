#!/bin/bash
# Wait for a pushed Kaggle kernel to finish, then download its output.
#
#   bash kaggle/wait_fetch.sh causalvis-vlm-smoke vlm_baseline/kaggle/outputs/smoke
#
# Only a finished state ends the wait; a lost connection is retried. The credential is read
# from ~/.kaggle, so no token appears here.
NAME=$1
DEST=${2:-/g/CausalVis/kaggle/outputs/$NAME}
USER=bukharihamzamukhtar
K=$(command -v kaggle || echo "/c/Users/Hamza Bukhari/AppData/Roaming/Python/Python314/Scripts/kaggle.exe")
SLUG="$USER/$NAME"

while true; do
  s=$("$K" kernels status "$SLUG" 2>&1 | tail -1)
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "[$(date +%T)] $NAME: $s"; break ;;
    *RUNNING*|*QUEUED*) sleep 180 ;;
    *) echo "[$(date +%T)] $NAME: no status (${s:0:90}), retrying"; sleep 120 ;;
  esac
done

mkdir -p "$DEST"
for try in 1 2 3 4 5; do
  "$K" kernels output "$SLUG" -p "$DEST" 2>&1 | tail -2
  ls "$DEST" | grep -q . && break
  echo "[$(date +%T)] download incomplete (try $try), retrying"; sleep 60
done
find "$DEST" -type f | head -10
