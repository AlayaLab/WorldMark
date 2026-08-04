#!/usr/bin/env bash
# =============================================================================
# WorldMark — unified evaluation driver (all 9 metrics, one command).
#
# Pipeline (each stage is idempotent + resumable + shardable across GPUs):
#   FLOW  -> Action Dynamics M1-M4 (SEA-RAFT optical flow + Depth-Anything-3)
#   QA    -> Visual Quality        (Q-Align / OneAlign: quality + aesthetics)
#   C3    -> World Memory: Global  (VGGT-Omega feed-forward reconstruction)
#   C1    -> World Memory: Local   (adjacent mutation; needs FLOW cache)
#   C2    -> World Memory: Revisit (equal-motion pairing; needs FLOW cache)
#   AGG   -> aggregate into master CSV + colored PNG
#
# The four compute stages use different, mutually-incompatible deep-learning
# environments (torch versions differ), so each stage is launched with its own
# Python interpreter, configured via the PY_* environment variables below.
#
# Configuration (export before running, or edit here):
#   WORLDMARK_VIDEOS   video root, layout {root}/{model}/{stem}.mp4   (required)
#   WORLDMARK_RESULTS  output root                                    (default ./results)
#   EVAL_SUFFIX         tag appended to result dirs (e.g. _real_first) (default empty)
#   PY_FLOW PY_QA PY_C3 PY_CONSISTENCY   per-stage python interpreters (default: python)
#   NGPU                number of GPUs to shard across                 (default 2)
# =============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$HERE/worldmark"
export PYTHONPATH="$PKG:${PYTHONPATH:-}"

PY_FLOW="${PY_FLOW:-python}"                 # env with SEA-RAFT + Depth-Anything-3
PY_QA="${PY_QA:-python}"                     # env with Q-Align / OneAlign
PY_C3="${PY_C3:-python}"                     # env with VGGT-Omega
PY_CONSISTENCY="${PY_CONSISTENCY:-python}"   # env with DINOv2 + LPIPS + TransNetV2
NGPU="${NGPU:-2}"
LOG="$HERE/logs"; mkdir -p "$LOG"

run_stage(){   # name  done_grep  interpreter  script  extra_env
  local name="$1" dg="$2" py="$3" script="$4" extra="$5"
  echo "$(date +%H:%M) === STAGE $name START ==="
  for att in 1 2 3; do
    local need=0
    for ((S=0; S<NGPU; S++)); do
      if ! grep -q "$dg" "$LOG/${name}_s$S.log" 2>/dev/null; then
        need=1
        env $extra SHARD=$S NSHARD=$NGPU CUDA_VISIBLE_DEVICES=$S "$py" "$PKG/$script" >> "$LOG/${name}_s$S.log" 2>&1 &
      fi
    done
    [ $need -eq 0 ] && break
    wait
    local ok=1
    for ((S=0; S<NGPU; S++)); do grep -q "$dg" "$LOG/${name}_s$S.log" 2>/dev/null || ok=0; done
    [ $ok -eq 1 ] && { echo "$(date +%H:%M) $name complete"; break; }
    echo "$(date +%H:%M) $name attempt $att incomplete, retrying"
  done
  echo "$(date +%H:%M) === STAGE $name END ==="
}

run_stage FLOW "DONE"             "$PY_FLOW"        flow_batch_full.py         ""
run_stage QA   "DONE qalign"      "$PY_QA"          qalign_batch.py            ""
run_stage C3   "DONE vggt"        "$PY_C3"          vggt_omega_global_c3.py    ""
run_stage C1   "DONE consistency" "$PY_CONSISTENCY" consistency_v2.py          "C1_ONLY=1"
run_stage C2   "DONE consistency" "$PY_CONSISTENCY" consistency_v2.py          "C2_ONLY=1"

echo "$(date +%H:%M) === AGGREGATE ==="
"$PY_CONSISTENCY" "$PKG/final_master.py"
"$PY_CONSISTENCY" "$PKG/render_master_table.py"
echo "$(date +%H:%M) === DONE: master table at $WORLDMARK_RESULTS ==="
