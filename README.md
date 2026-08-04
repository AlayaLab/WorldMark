# WorldMark

**[Project Page](https://alayalab.github.io/WorldMark/)** ·
**[Paper](https://arxiv.org/abs/2604.21686)** ·
**[World Model Arena](https://warena.ai/)**

A benchmark for **interactive world models** that scores generated videos along
three complementary axes — **Action Dynamics**, **World Memory**, and
**Visual Quality** — for a total of **9 metrics**. All metrics are
**deterministic and reproducible** (no SLAM randomness, no LLM sampling).

<img src="docs/static/images/teaser.png" width="50%" alt="WorldMark overview: per-model adapters translate one shared action vocabulary into each model's native control format; a round-trip probe compares outbound and return views at equal accumulated motion.">

Given a video the model produced in response to a scripted action sequence
(`W` forward / `S` back / `A` strafe-left / `D` strafe-right / `L` yaw-left /
`R` yaw-right, 20 s per key), WorldMark asks:

| Category | Question | Metrics |
|---|---|---|
| **Action Dynamics** | *When told to move, does it move correctly?* | Direction Accuracy · Direction Purity · Motion Stability · Response Latency |
| **World Memory** | *Does it remember the world it generated?* | Local Memory · Global Memory · Revisit Memory |
| **Visual Quality** | *Does each frame look good?* | Perceptual Quality · Aesthetic Quality |

`Direction Accuracy`, `Direction Purity`, `Motion Stability`, `Response Latency`
are reported split into **trans** (translation, W/S/A/D) and **rot** (rotation,
L/R); `Revisit Memory` pools translation + rotation round-trips. In the master
table every value is normalized to **[0, 100], higher is better**.

See [`docs/METHODS.md`](docs/METHODS.md) for motivation, ablations, and exact formulas.

---

## What each metric measures

**Action Dynamics** — flow + depth based (SEA-RAFT optical flow, Depth-Anything-3 depth):
- **Direction Accuracy** — is the induced motion in the commanded direction (sign of net vs total flow, weighted by depth-parallax motion type).
- **Direction Purity** — is the motion confined to the commanded axis, or does it leak into other axes.
- **Motion Stability** — once moving, does the motion stay steady (no hiccups / decay).
- **Response Latency** — how promptly the model responds to a command change.

**World Memory**:
- **Local Memory** — no abrupt frame-to-frame mutations / flicker / hard cuts (adjacent, short timescale). DINOv2 patch-mean + TransNetV2.
- **Revisit Memory** — leave a viewpoint and return to the *same pose*: is the world still there or re-generated (equal-motion frame pairing + DINOv2).
- **Global Memory** — is the whole recovered trajectory + point cloud globally consistent (VGGT-Omega feed-forward reconstruction).

**Visual Quality** — Q-Align / OneAlign (a multimodal LLM, logit-readout, deterministic):
- **Perceptual Quality** (video quality / VQA) and **Aesthetic Quality** (IAA).

---

## Install

The four compute stages use **incompatible torch stacks**, so create **one
virtualenv per stage** and install the matching requirements file:

```bash
python -m venv envs/flow        && envs/flow/bin/pip        install -r requirements/flow.txt
python -m venv envs/qalign      && envs/qalign/bin/pip      install -r requirements/qalign.txt
python -m venv envs/consistency && envs/consistency/bin/pip install -r requirements/consistency.txt
python -m venv envs/c3          && envs/c3/bin/pip          install -r requirements/c3.txt
```

External repos to clone and point at via env vars (see `worldmark/wm_config.py`):
- **SEA-RAFT** → export `SEARAFT_CFG=/path/to/SEA-RAFT/config/eval/spring-M.json`
- **Depth-Anything-3** → `DA3_MODEL` (HF id or local path, default `depth-anything/DA3-LARGE-1.1`)
- **VGGT-Omega** → `VGGT_OMEGA_PATH=/path/to/vggt-omega`
- **Q-Align** → `QALIGN_MODEL` (default `q-future/one-align`)

---

## Data layout

```
$WORLDMARK_VIDEOS/
  <ModelName>/
    000_006.mp4        # stem "aaa_bbb": bbb = action id (here 6 = W→S round trip)
    001_007.mp4
    ...
```
Action ids and their key sequences are defined in `worldmark/protocol.py`.

---

## Run

```bash
export WORLDMARK_VIDEOS=/path/to/videos          # {model}/{stem}.mp4
export WORLDMARK_RESULTS=/path/to/out
export EVAL_SUFFIX=_real_first                     # keeps domains/views separate
export PY_FLOW=envs/flow/bin/python
export PY_QA=envs/qalign/bin/python
export PY_C3=envs/c3/bin/python
export PY_CONSISTENCY=envs/consistency/bin/python
# export NGPU=2   # optional: shard across N GPUs (default 1)

bash run_eval.sh
```

This runs all stages on a single GPU by default (each idempotent, resumable,
and shardable across `NGPU` GPUs),
then writes:
- `$WORLDMARK_RESULTS/results_quality$EVAL_SUFFIX/final_master.csv`
- `$WORLDMARK_RESULTS/master_table_<tag>.png` (colored: green = 1st, yellow = 2nd, red = worst per column)

Per-video raw scores are kept under `results_*` (JSON) so you can re-aggregate or
change the reporting normalization without recomputing.

### Run a single stage
Each stage is a standalone script under `worldmark/`, driven by
`SHARD` / `NSHARD` / `CUDA_VISIBLE_DEVICES` and the `wm_config.py` paths:
```bash
PYTHONPATH=worldmark CUDA_VISIBLE_DEVICES=0 $PY_FLOW worldmark/flow_batch_full.py
# multi-GPU: run one process per shard, e.g. SHARD=0 NSHARD=2 / SHARD=1 NSHARD=2
```

---

## Cost & reproducibility

- **~1.5 h per 125-video set** on a single H200 for all 9 metrics, dominated by
  the M1-M4 flow stage; sharding scales linearly (~45 min on two GPUs via
  `NGPU=2`). With the flow cache present, a re-score takes a fraction of that.
- All metrics are deterministic: Q-Align and VGGT-Omega are bit-identical
  run-to-run; SLAM back-ends (DROID-SLAM, ViPE) were deliberately **not** used for
  Global Memory because they are non-reproducible.

## Repository layout

```
run_eval.sh                 unified driver (all stages)
requirements/               per-stage dependency pins
worldmark/
  wm_config.py                all paths via env vars (edit or export)
  protocol.py               action-id -> key-sequence / segment boundaries
  flow_eval.py              M1-M4 perceive + score (SEA-RAFT + DA3)
  flow_batch_full.py        M1-M4 batch driver
  dir_metric2.py            pose-based direction helper
  consistency_v2.py         C1 Local + C2 Revisit
  vggt_omega_global_c3.py   C3 Global
  qalign_batch.py           Visual Quality (Q-Align)
  final_master.py           aggregate -> master CSV
  render_master_table.py    master CSV -> colored PNG
docs/METHODS.md             design rationale + formulas
```

## License
See `LICENSE`.
