#!/bin/bash
# Push one built kernel to Kaggle as soon as a session is free, wait for it, and download its
# output into kaggle/outputs/<kernel>.
#
#   KAGGLE_API_TOKEN=... bash kaggle/run_kaggle.sh causalvis-train-seed3
#   KAGGLE_API_TOKEN=... bash kaggle/run_kaggle.sh causalvis-train-seed4 43200
#
# The token is read from the environment only, so it never lands in a file here.
NAME=$1
T=${2:-43200}
USER=${KAGGLE_USER:-bukharihamzamukhtar}
K=${KAGGLE:-kaggle}      # set KAGGLE if the command is not on the PATH
BUILD="/g/CausalVis/kaggle/build/kernels/$NAME"
DEST="/g/CausalVis/kaggle/outputs/$NAME"
SLUG="$USER/$NAME"

[ -d "$BUILD" ] || { echo "no kernel folder $BUILD (run prepare_seeds.py first)"; exit 1; }

while true; do
  r=$("$K" kernels push -p "$BUILD" -t "$T" 2>&1 | tail -1)
  case "$r" in
    *successfully*) echo "[$(date +%T)] pushed $NAME"; break ;;
    *) echo "[$(date +%T)] not pushed yet: ${r:0:90}"; sleep 300 ;;
  esac
done

while true; do
  s=$("$K" kernels status "$SLUG" 2>&1 | tail -1)
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "[$(date +%T)] $NAME: $s"; break ;;
    *RUNNING*|*QUEUED*) sleep 300 ;;
    *) echo "[$(date +%T)] $NAME: no status (${s:0:90}), retrying"; sleep 120 ;;
  esac
done

mkdir -p "$DEST"
for try in 1 2 3 4 5; do
  "$K" kernels output "$SLUG" -p "$DEST" 2>&1 | tail -2
  ls "$DEST" | grep -q . && break
  echo "[$(date +%T)] download incomplete (try $try), retrying"; sleep 120
done
echo "files in $DEST:"; ls -la "$DEST" | tail -5
