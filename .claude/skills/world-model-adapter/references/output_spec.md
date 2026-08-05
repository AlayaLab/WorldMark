# I/O spec

The contract. Everything here is fixed; how you satisfy it for a given repo is up to you.

---

## 1. Input: `arena_inputs/`

Unpacked from `arena_inputs.zip`. **Read-only — never edit.**

```
arena_inputs/
  README.txt                      # authoritative description
  action_protocol.txt             # action_id -> key sequence (1..15)
  first_view/
    real/NNN.jpg                          # 000..024, 25 first-person images
    real_intrinsics/NNN_intrinsics.npy    # (4,) float32 [fx, fy, cx, cy]
    real_action.txt                       # row i = image i's 5 assigned action_ids
    prompt_real.txt                       # row i = image i's caption (25 rows)
    style/  style_intrinsics/  style_action.txt  prompt_style.txt
  third_view/
    real/  real_intrinsics/  real_action.txt  prompt_real.txt
    style/ style_intrinsics/ style_action.txt  prompt_style.txt
```

> `README.txt` matches this layout as shipped. Still trust the filesystem over any doc
> (trap 12), and re-check both if the inputs are revised.

Dimensions: **view** ∈ {`first_view`, `third_view`} × **domain** ∈ {`real`, `style`}.
Per (model × view × domain): 25 images × 5 actions = **125 videos**.
All four combinations: **500 videos per model**.

Facts verified against the shipped data — do not re-derive, but do re-check if the zip changes:

- `*_action.txt` is **identical between first_view and third_view** for a given domain.
- All four `prompt_*.txt` exist, 25 rows each, and **differ between views**: every
  `first_view` caption starts with `First-person view.` and every `third_view` caption with
  `Third-person view.` — the prompt already carries the view signal. See §5.
- Intrinsics are **per-image and all distinct**, in **original-image pixel units**
  (`cx ≈ (W-1)/2`, `cy ≈ (H-1)/2`). See §4 — they must be transformed, not passed through.
- Image sizes are **not uniform**: `real` spans 1024×768 … 1920×1200 (AR 1.33–1.79);
  `style` is uniformly 1024×1024 (AR 1.00).
- Real horizontal FOV spans **22°–90°** (median ~66° for `real`, ~45° for `style`).

### Action protocol

`W`=forward `S`=back `A`=strafe-left `D`=strafe-right `L`=turn-left `R`=turn-right

| id | keys | id | keys | id | keys |
|----|------|----|------|----|------|
| 1 | W | 6 | WS | 11 | WSW |
| 2 | S | 7 | LR | 12 | LRL |
| 3 | A | 8 | AD | 13 | WRW |
| 4 | D | 9 | WR | 14 | ADA |
| 5 | R | 10 | SL | 15 | WRS |

Each key-segment is held **~20 s** (16 fps reference):

| segments | action ids | target duration | reference frames @16fps |
|----------|-----------|-----------------|------------------------|
| 1 | 1–5 | ~20 s | ~320 |
| 2 | 6–10 | ~40 s | ~640 |
| 3 | 11–15 | ~60 s | ~960 |

If the target model cannot express a key natively, **fail loudly for that sample** — never
silently substitute a different motion.

---

## 2. Output: one mp4 per (image × action)

**The mp4 is the only required deliverable.** Evaluation extracts poses, depth and flow from
the video itself; the model does not emit any auxiliary geometry.

### Filename

```
{image:03d}_{action_id:03d}.mp4        e.g.  016_014.mp4   # image 016, action 14 (ADA)
```

### Directory — produce both layouts

```
# canonical
{domain}/{view}/{MODEL}/{image}_{action}.mp4
    e.g.  real/first/ALAYAWORLD/000_006.mp4

# flat
{RELEASE}_video_all/{MODEL}_{view}_{domain}/{image}_{action}.mp4
    e.g.  0721_video_all/ALAYAWORLD_first_real/016_014.mp4
```

`view` is shortened to `first` / `third` in output paths. `MODEL` is the upper-case model tag
and must be stable across runs. Generate one layout and hard-link/symlink the other rather
than rendering twice.

### Acceptance

`generation/check_delivery.py` is the mechanical gate for everything in this section — filenames,
completeness against `{view}/{domain}_action.txt`, decodability, and duration tolerance. It
decodes the videos and **does not read the manifest**, so it is the arbiter, not a formality:

```bash
python generation/check_delivery.py \
    --arena arena_inputs/ --delivery real/first/<MODEL> --view first_view --domain real \
    [--tolerance 10] [--probe-backend auto|ffprobe|cv2] [--json-out report.json]
```

Exit 0 = accepted. Anything else is not a delivery.

### Video spec — native, not unified

**Do not force a common resolution or fps.** Each model keeps its native output; the
evaluation side resizes (to 432×640 for flow) and time-normalises signals. Achieved values
therefore differ per model and that is expected — for example, observed across models:
960×544@19fps, 1280×704@17fps, 1280×704@16fps, 768×512@16fps, 832×480@16fps.

What you **must** do is **measure and report** the actual resolution, fps, frame count and
duration of what you produced (§3). Never report the nominal target as if it were achieved.

---

## 3. Provenance: `manifest.jsonl` (recommended, not consumed by eval)

Evaluation needs only the mp4s. Keep a manifest anyway — it is what lets you answer "what
exactly did model X receive" months later. One JSON object per line, per video:

```json
{"model": "ALAYAWORLD", "view": "first", "domain": "real",
 "image": "000", "action_id": 6, "keys": "WS",
 "video": "real/first/ALAYAWORLD/000_006.mp4",

 "native": {"format": "camera_pt", "value": null, "files": ["000_006_camera.pt"]},

 "requested": {"segments": 2, "target_sec": 40.0},
 "achieved":  {"width": 960, "height": 544, "fps": 19, "fps_source": "writer_call",
               "frames": 1145, "sec": 60.26, "measured": true},
 "segments": [{"keys": "W", "frames": [0, 572]}, {"keys": "S", "frames": [572, 1145]}],

 "intrinsics": {"supplied": true, "transform": "scale+centercrop to 960x544",
                "src_fov_deg": 68.2, "model_fov_deg": 68.2},
 "prompt": {"source": "third_view/prompt_real.txt:1"},
 "caveats": ["absolute speed stripped by relative-pose normalisation"]}
```

Rules for the fields that matter:

- `achieved` is **measured from the produced file** (`ffprobe`/decoder), never copied from
  the request. `measured: false` means unchecked.
- `fps_source` ∈ `writer_call` / `config` / `readme` — say which value governs wall-clock
  duration. These disagree in real repos.
- `segments[].frames` are half-open indices **into the produced video**, so evaluation can cut
  per key-segment without trusting the nominal 20 s.
- `intrinsics.transform` records what you did to the supplied `[fx,fy,cx,cy]` (§4).
  If the model ignores intrinsics, set `supplied: false` and put the model's own FOV in
  `model_fov_deg` so the mismatch is visible.
- `caveats` carries cross-model comparability warnings (scale normalised away, eased
  boundaries, >360° accumulated turn, approximated key).

---

## 4. Intrinsics — four cases, only one consumes ours

First classify the model. **Only case 4 uses the `*_intrinsics/` files we ship**; in cases
1–3 we supply nothing and the adapter must not invent an injection point.

| # | Model behaviour | Uses our intrinsics? | Adapter action |
|---|---|---|---|
| 1 | **No camera model** — conditioned only on action/key vectors | no | Ignore intrinsics entirely. Set `supplied: false`, `transform: "n/a (no camera model)"`. |
| 2 | **Hardcoded, or derived from output resolution** — virtual pinhole: no distortion, principal point centred | no | Pass nothing. Record the model's implied FOV next to the source FOV so the mismatch is auditable. Do **not** patch the model's constant unless the user asks. |
| 3 | **Estimated per-image at runtime** by the model's own estimator | no | Let it estimate. Record `supplied: false`, the estimator, and any accept/reject range. |
| 4 | **Camera-conditioned, requires user-supplied intrinsics** | **yes** | Transform ours to the model's working resolution (below), pass them, record the transform. |

### Case 4 only: the transform

Our `[fx, fy, cx, cy]` are in **original-image pixels**. The model resizes (and often changes
aspect ratio) first, so raw pass-through is wrong. Apply the transform matching the model's
own resize/crop:

```
scale:       fx' = fx * sx      fy' = fy * sy      cx' = cx * sx      cy' = cy * sy
crop(x0,y0): cx' = cx - x0      cy' = cy - y0
```

Some repos expect intrinsics **normalised** by the working resolution, or relative to a fixed
reference resolution rather than the actual one — check which, with `file:line`, before
scaling.

### Aspect ratio (all cases)

`style` images are square (AR 1.00), `real` spans 1.33–1.79, and most models generate ~1.7–1.8.
Choose **one** policy per model (center-crop / pad / stretch), apply it to every sample, record
it in `README.md`, and — under case 4 — reflect it in the intrinsics transform.

---

## 5. Prompt policy

Every (view, domain) supplies one caption per image: `{view}/prompt_{domain}.txt`, row `i` for
image `NNN=i`. Use them verbatim.

- **Never cross views or domains.** `first_view/prompt_real.txt:5` belongs to
  `first_view/real/005.jpg` and nothing else. The captions differ between views by design —
  each is prefixed `First-person view.` / `Third-person view.`, so a mismatched prompt tells
  the model the wrong camera setup.
- **Do not rewrite, truncate, or "improve" captions.** If the model has a token limit that the
  caption exceeds, truncate deterministically (e.g. first N chars), apply the identical rule to
  all 25 images, and record it in `README.md` + the manifest.
- The view prefix is part of the caption. Strip it only if the model has its own explicit
  first/third-person control and the prefix demonstrably conflicts — and then say so.
- If a model takes **no** text input, ignore the prompt files; set `prompt.source: null`.
- If a model wants a negative prompt, use its own documented default, uniformly.

Record `prompt.source` as `"{view}/prompt_{domain}.txt:{row}"` so any caption in a rendered
video can be traced back.

---

## 6. Deliverables per model

```
adapters/<model_id>/
  PROBE.md          # filled probe_checklist, file:line evidence
  adapt.py          # CPU-only: arena_inputs + (view,domain) -> native inputs
  run_batch.py      # GPU: load model once, loop 125 samples, resumable
  test_golden.py    # asserts the canonical conversions (no GPU)
  patches/*.patch   # repo edits, git-diff format  (see integration levels below)
  README.md         # integration level, exact launch command, env notes,
                    # prompt / aspect-ratio / intrinsics policy
```

### Integration levels — not every repo can be driven from outside

`adapt.py` writes native inputs **where a native input surface exists**. Often it does not.
Classify the repo, state the level in `PROBE.md` and `README.md`, and expect to patch:

| Level | Situation | What `adapt.py` writes | Repo changes |
|-------|-----------|------------------------|--------------|
| **A** | A file or CLI control surface already exists and is batchable | the native files / an args JSON | none, or bug fixes only |
| **B** | Control input exists but is **interactive-only** or otherwise not batchable (e.g. a `input()` prompt per chunk) | native files that only the **patched** repo can read | a patch that **adds** a file/flag input path, feeding the same internal call the interactive path fed |
| **C** | Per-sample inference is not separable from model loading, or the entry point is hardwired to one sample | same as B | patch + `run_batch.py` imports the pipeline class directly instead of shelling out to the repo's CLI |

Levels B and C are normal, not failures. Matrix-Game-class repos are level B: the only control
surface was an interactive stdin loop, so a `--trajectory_file` flag had to be added before any
batch run was possible.

### Patch discipline

**A patch may change how control input gets in. It must not change what the model does with
it.** Rewriting a speed constant, a coordinate convention, an intrinsics default, or the
conditioning path silently re-defines the model and invalidates cross-model comparison.

Allowed without asking:
- adding a flag / file loader that constructs the *same* control object the existing path built;
- removing `exit()` / `destroy_process_group()` from a per-sample function so a loop survives;
- fixing an outright upstream bug that prevents the run (wrong kwarg name, broken import).

Requires asking the user first:
- changing any magnitude, FOV, fps, or coordinate constant;
- changing the conditioning mechanism or which branch of a two-path repo runs;
- upgrading/downgrading a dependency that alters numerics.

Every patch: one `.patch` file per concern, `git diff` format, a one-line rationale at the top,
and a note in `PROBE.md` F5. `run_batch.py` must verify patches are applied before starting
(`git diff --stat` or a marker check) rather than assuming.

`adapt.py` must run **without GPU and without the model's env** (numpy/scipy-level deps), so
plans can be reviewed and diffed before spending GPU hours:

```bash
python adapt.py --arena arena_inputs/ --view first_view --domain real \
                --out plans/<model_id>_first_real/ [--dry-run]
```

`--dry-run` prints the per-action timing table (segments, units, frames, seconds, deviation)
and writes nothing.

**Determinism**: same input + same adapter version → byte-identical native inputs. No
wall-clock, no unseeded randomness.

### `run_batch.py`

- Load model/VAE/text-encoder **once**; loop samples.
- `--resume` default on: skip samples whose mp4 already exists.
- Append per-sample status to `jobs.jsonl` on disk. Progress must survive process death.
- Reset per-sample global state found in probe F3 (KV/VAE caches, RNG).
- Per-sample failure: log, mark failed, **continue**.
- After each sample, probe the produced mp4 and fill `achieved`.

### `test_golden.py`

No GPU. Must fail if mapping or timing changes. Assert for one sample of each segment count
(e.g. action 1 / 6 / 11):

- native value or file contents exactly (string equality, or array shape + sampled values)
- per-segment frame boundaries and total frames
- pose-based models: forward-axis position monotone during `W`; accumulated yaw monotone and
  translation ~0 during `R`; gravity-axis component ~0 if the convention says so
- intrinsics transform arithmetic against a known image size

Property assertions catch axis and unit errors. They do **not** catch a globally flipped sign
— that is gate G3.

---

## 7. Duration rounding

When 20 s is not an integer number of the model's rollout units:

1. `N = round(20 / unit_sec)`, the **same N for every segment** of a sample.
2. The same `N` for every sample of a given (model, view, domain).
3. Record real seconds and deviation in `achieved`; never let downstream assume 20 s.

Evaluation time-normalises signals, so a few percent deviation is acceptable — **uniformity
and honest reporting matter more than hitting 20.0 s.**

Exceptions:
- a model whose first chunk differs in length cannot have equal segments; accept it, record
  both lengths in `segments`, note it in `caveats`;
- for long rollouts where per-segment rounding drifts badly, quantise cumulative boundaries
  (`boundary_k = round(k * 20 / unit_sec)`), trading uniformity for bounded total error; note
  the choice.
