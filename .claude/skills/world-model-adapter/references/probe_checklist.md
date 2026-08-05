# Probe checklist

Copy this into `adapters/<model_id>/PROBE.md` and fill it in. Every answer needs `file:line`
evidence, or the literal token `UNVERIFIED`. An answer without evidence is a defect.

Header:

```
model_id:      <short_snake_case>
repo:          <path or URL>
commit:        <git rev-parse HEAD>
probed_by:     <date>
```

---

## A. Entry point

- **A1** Which script/function generates one sample? `file:line`
- **A2** List *every* control-related CLI arg or input file, with the argparse/loader line.
- **A3** Are there multiple inference paths (interactive vs batch, base vs distilled,
  streaming vs full)? Do they take control input differently? `file:line` for each.
  → If semantics differ, say which one this adapter targets and why.
- **A4** Required input files besides the image (prompt file, intrinsics, camera file)?
  What happens if one is missing?

## B. Action representation

- **B1** Native control primitive — pick one and cite it:
  per-frame key/mouse vectors · letter DSL string · pose matrices · text captions ·
  something else
- **B2** If letters/keys: **paste the mapping dict verbatim** with `file:line`.
  Do not summarise. Do not assume `w/a/s/d` are translations.
- **B3** If pose matrices:
  - c2w or w2c? Cite the *consumer*, not the README.
  - Coordinate convention (axis directions, handedness). How did you determine it?
    (e.g. which component the code zeroes when projecting motion to the ground plane;
    `det(R)` sign; an explicit comment.)
  - Array shape and dtype, measured on a real file if the repo ships an example.
- **B4** If text: paste the exact template and the full vocabulary, with `file:line`.
- **B5** Are simultaneous/composite actions supported (move + turn at once)? How encoded?
  (The 6 benchmark keys are all single-axis, but knowing this tells you whether the DSL
  needs one token per key or can combine them. Key mapping itself: see H4.)

## C. Time

- **C1** What unit does the user-facing duration take — pixel frames, latents, chunks,
  fixed-length actions, or text lines? Give the multiplier to pixel frames with `file:line`.
- **C2** Output fps. Check **both** the config/constant **and** the actual video-writer call.
  If they disagree, record both and say which one governs wall-clock duration.
- **C3** Length constraints (`4n+1`, `1+80k`, multiple of chunk, ...)? Where enforced?
- **C4** Is the first chunk/clip a different length from subsequent ones? `file:line`
- **C5** Compute and record: for a 20 s target, the integer unit count you will use, the
  resulting frames, the resulting seconds, and the % deviation. (Rounding policy:
  see `output_spec.md`.)

## D. Magnitude

- **D1** Translation step per frame — the constant, its units, `file:line`, **plus any later
  rescale** (a `*0.01` twenty lines away changes the meaning entirely).
- **D2** Rotation step per frame — constant + `file:line`.
- **D3** Is motion magnitude normalised, made relative, or otherwise scale-stripped before
  it reaches the network? Grep `normalize`, `relative_pose`, `/ max`, `norm(`.
  → If yes: **absolute speed is not controllable**; record this, it changes what the
  benchmark can claim.
- **D4** Is motion step-constant, or eased/smoothed (inertia, exponential ramp)?
  If eased, give the time constants — segment boundaries are then not sharp.
- **D5** Accumulated rotation for a 20 s turn = rotation step × frames. Record the number.
  If it exceeds 360°, say so.

## E. Intrinsics / geometry

- **E1** Classify into exactly one of the four cases in `output_spec.md §4`, with `file:line`:
  1. no camera model at all · 2. hardcoded or derived from resolution ·
  3. estimated at runtime · 4. requires user-supplied intrinsics
  → **Only case 4 consumes `arena_inputs/*_intrinsics/`.** In cases 1–3 the adapter must not
  invent an injection point.
- **E2** Case 2 — the values, whether pixels or normalised, and the implied horizontal FOV.
  (Our data spans 22°–90°; record the gap.)
- **E3** Case 4 — required shape/dtype/layout, pixels vs normalised, and **which** resolution
  any internal rescale assumes (the actual working resolution, or a fixed reference?). Cite it.
- **E4** Case 3 — which estimator, and any accept/reject range.
- **E5** Output resolution, and how the input image gets there: resize / center-crop / pad /
  stretch. `file:line`. Our images span AR 1.00–1.79, so this is never a no-op.

## F. Batch-readiness

- **F1** Does the per-sample path call `exit()`, `sys.exit()`, or `destroy_process_group()`?
  `file:line`. These make a loop terminate after one sample.
- **F2** Can model loading be separated from per-sample inference? Which object is reusable?
- **F3** Per-sample global state that must be reset between samples — VAE/feature caches,
  KV cache, RNG, `torch.compile` shape specialisation. `file:line`
- **F4** Multi-GPU: what launcher and flags? Anything that breaks when looping
  (e.g. process group torn down inside the sample function)?
- **F5** Any known upstream bug hit during the probe? Record it and put the fix in
  `patches/`.
- **F6** **Integration level** (A / B / C, see `output_spec.md §6`), with the evidence:
  - **A** a batchable file or CLI control surface already exists → cite it;
  - **B** control input exists but is interactive-only / unbatchable → cite the `input()` or
    equivalent, and name the internal call your patch will feed instead;
  - **C** per-sample inference not separable from model loading → cite what blocks it, and name
    the class `run_batch.py` will import directly.
  List every patch you intend to write and, for each, confirm it changes only *how input gets
  in* — not any magnitude, constant, or conditioning path. Anything else needs the user's
  approval first.

## G. Injection mechanism *(documentation only — optional)*

- **G1** How does the control signal enter the network — extra input channels, additive
  embedding, cross-attention, attention position encoding? `file:line`
  Useful for the paper; not needed to render. Mechanisms are **not mutually exclusive** (see
  trap 11). Write `UNVERIFIED` rather than guessing.

## H. Suite fit

- **H1** Does the model take a text prompt? `file:line`. Any length/token limit? If the
  supplied captions exceed it, state the deterministic truncation rule applied to all 25.
  Does the model have its own first/third-person control that could conflict with the
  caption's view prefix?
- **H2** Does the model distinguish first- vs third-person, via a flag, a checkpoint, or not
  at all? `file:line` or `UNVERIFIED`.
- **H3** Can the model start from an arbitrary user image, or does it expect a specific
  preprocessing/format (square, fixed size, specific colour space)?
- **H4** Which of the 6 benchmark keys (`W S A D L R`) have exact native equivalents, and which
  require approximation? Table with the native token for each. Flag every approximation.

---

## Sign-off

```
Integration level: A / B / C      Patches: <list, or none>
Intrinsics case: 1 / 2 / 3 / 4
Prompt handling: <verbatim | truncated to N chars | ignored (no text input)>
Aspect-ratio policy: <center-crop | pad | stretch>
Key mapping: W-> S-> A-> D-> L-> R->        (mark approximations)

Timing: 1-seg = __ frames = __ s ;  2-seg = __ ;  3-seg = __   (deviation __ %)
Achieved (measured): __ x __ @ __ fps, fps_source = __

Gates:  G0 [ ]  G1 [ ]  G2 [ ]  G3 [ ]  G4 [ ]
UNVERIFIED items: <list, or none>
Cross-model caveats: <e.g. absolute speed stripped; eased boundaries; >360 deg turn;
                      hardcoded FOV vs source FOV gap>
```
