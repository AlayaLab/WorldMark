# Evaluation

Scores a delivery of benchmark videos on the 9 WorldMark metrics. Every estimator is
feed-forward, so results are **bit-identical across runs**.

What each metric measures, and the formulas: **[`../docs/METRICS.md`](../docs/METRICS.md)**.

| Category | Metrics |
|---|---|
| **Action Dynamics** | Direction Accuracy · Direction Purity · Motion Stability · Response Latency |
| **World Memory** | Local · Global · Revisit |
| **Visual Quality** | Perceptual · Aesthetic |

The four Action Dynamics metrics are reported split into **trans** (W/S/A/D) and **rot** (L/R);
Revisit Memory pools both. Every value is normalized to **[0, 100], higher is better**.

---

## Before you start

Validate the delivery first — scoring a malformed set silently produces wrong numbers:

```bash
python ../generation/check_delivery.py \
    --arena ../arena_inputs/ --delivery /path/to/real/first/MYMODEL \
    --view first_view --domain real
```

---

## Install

All four stages share **one** environment:

```bash
python -m venv envs/worldmark
envs/worldmark/bin/pip install -r requirements.txt
```

Tested on Python 3.10/3.11 with CUDA 12.1 wheels. The binding pin is
`transformers==4.36.1` — Q-Align's mPLUG-Owl2 breaks above it, and that version still serves
`facebook/dinov2-base`. `torch==2.4.1` clears VGGT-Omega's `>=2.4` floor while remaining fine
for Q-Align, SEA-RAFT, DINOv2, LPIPS and TransNetV2. See the comments in
[`requirements.txt`](requirements.txt) for the install-order gotchas.

External repos to clone and point at via env vars (see `worldmark/wm_config.py`):

| Env var | What | Default |
|---|---|---|
| `SEARAFT_CFG` | [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) eval config json | — |
| `DA3_MODEL` | [Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-3) | `depth-anything/DA3-LARGE-1.1` |
| `VGGT_OMEGA_PATH` | VGGT-Omega checkout | — |
| `QALIGN_MODEL` | Q-Align / OneAlign | `q-future/one-align` |

---

## Data layout

```
$WORLDMARK_VIDEOS/
  <MODEL>/
    000_006.mp4        # stem "aaa_bbb": bbb = action id (6 = W->S round trip)
    016_014.mp4
    ...
```

Action ids and their key sequences are defined in `worldmark/protocol.py` and
`../arena_inputs/action_protocol.txt`.

---

## Run

```bash
source envs/worldmark/bin/activate
export WORLDMARK_VIDEOS=/path/to/real/first     # the dir containing <MODEL>/*.mp4
export WORLDMARK_RESULTS=/path/to/out
export EVAL_SUFFIX=_real_first                  # keeps domains/views separate
# export NGPU=2                                 # optional, default 1

bash run_eval.sh
```

`run_eval.sh` calls plain `python`, so activating the env is all it needs. If you ever do have
to split a stage out (e.g. a future component needs `transformers>=4.40` and you keep Q-Align
separate), point `PY_FLOW` / `PY_QA` / `PY_C3` / `PY_CONSISTENCY` at the other interpreter.

Every stage is idempotent and resumable, so an interrupted run picks up where it stopped.
Outputs:

- `$WORLDMARK_RESULTS/results_quality$EVAL_SUFFIX/final_master.csv`
- `$WORLDMARK_RESULTS/master_table_<tag>.png` — green = 1st, yellow = 2nd, red = worst per column

Per-video raw scores stay under `results_*` as JSON, so you can re-aggregate or change the
reporting normalization without recomputing.

### A single stage

```bash
PYTHONPATH=worldmark CUDA_VISIBLE_DEVICES=0 python worldmark/flow_batch_full.py
# multi-GPU: one process per shard, e.g. SHARD=0 NSHARD=2 / SHARD=1 NSHARD=2
```

| Stage | Script | Produces |
|---|---|---|
| FLOW | `flow_batch_full.py` → `flow_eval.py` | Action Dynamics (SEA-RAFT + Depth-Anything-3) |
| QA | `qalign_batch.py` | Visual Quality (Q-Align) |
| C1/C2 | `consistency_v2.py` | Local + Revisit Memory (DINOv2 + TransNetV2) |
| C3 | `vggt_omega_global_c3.py` | Global Memory (VGGT-Omega) |
| AGG | `final_master.py` → `render_master_table.py` | master CSV + table PNG |

---

## Cost

**~1.5 h per 125-video set** on a single H200 for all nine metrics, dominated by the flow
stage; sharding scales linearly (~45 min on two GPUs via `NGPU=2`). With the flow cache
present, a re-score takes a fraction of that.

SLAM back-ends (DROID-SLAM, ViPE) were deliberately **not** used for Global Memory: their
bundle adjustment is not deterministic, so scores would not reproduce.
