#!/bin/bash
# Push one VLM job to Kaggle as soon as a GPU session is free (two run at a time), wait for
# it, and download its answers into vlm_baseline/kaggle/outputs/<job>.
#
#   KAGGLE_API_TOKEN=... bash vlm_baseline/run_kaggle.sh smoke
#   KAGGLE_API_TOKEN=... bash vlm_baseline/run_kaggle.sh test 43200
#
# The token is read from the environment only, so it never lands in a file here.
JOB=$1
T=${2:-43200}
USER=${KAGGLE_USER:-bukharihamzamukhtar}
K=${KAGGLE:-kaggle}      # set KAGGLE if the command is not on the PATH
SLUG="$USER/causalvis-vlm-$JOB"
BUILD="/g/CausalVis/vlm_baseline/kaggle/build/kernels/$JOB"
DEST="/g/CausalVis/vlm_baseline/kaggle/outputs/$JOB"

while true; do
  r=$("$K" kernels push -p "$BUILD" -t "$T" 2>&1 | tail -1)
  case "$r" in
    *successfully*) echo "[$(date +%T)] pushed $JOB"; break ;;
    *) echo "[$(date +%T)] not pushed yet: ${r:0:90}"; sleep 300 ;;
  esac
done

while true; do
  s=$("$K" kernels status "$SLUG" 2>&1 | tail -1)
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "[$(date +%T)] $JOB: $s"; break ;;
    *RUNNING*|*QUEUED*) sleep 300 ;;
    *) echo "[$(date +%T)] $JOB: no status (${s:0:90}), retrying"; sleep 120 ;;
  esac
done

mkdir -p "$DEST"
for try in 1 2 3 4 5; do
  "$K" kernels output "$SLUG" -p "$DEST" 2>&1 | tail -2
  [ -s "$DEST/out/$JOB/answers.jsonl" ] && break
  echo "[$(date +%T)] download incomplete (try $try), retrying"; sleep 120
done
if [ -s "$DEST/out/$JOB/answers.jsonl" ]; then
  echo "FETCHED $JOB: $(wc -l < "$DEST/out/$JOB/answers.jsonl") answers"
else
  echo "FETCH FAILED $JOB"
fi
