# Known traps

Every entry below is a mistake that has actually been made, or a hazard found by reading
source that contradicted a reasonable assumption. Each maps to a checklist item.
Read this before concluding the probe.

---

## 1. Letter semantics are not portable → B2

The same letter means different things across repos. Observed: one repo maps `a`/`d` to
**yaw rotation** and `j`/`l` to **strafe translation**; several others do the exact opposite.
Another repo uses `a`/`d` for translation and spells rotation as `left_rot`/`right_rot`.

An adapter that copies a letter string from a sibling repo produces a video that moves when
it should turn — and passes every numeric test.

**Always grep the key→action dict and paste it into `PROBE.md`.**

## 2. A parameter named "speed" may not be a speed → C1, D1

Observed: a repo whose JSON takes `action_speed_list: [4, 6]` alongside `action_seq`.
Reading the consumer showed those numbers are **relative frame-count weights**
(`weight / total_weight * num_frames`); the actual motion magnitude came from a default
keyword argument elsewhere in the call chain.

**Trace every control number to the function that consumes it.** The same repo had a second
entry point where a similarly-named argument *was* a magnitude — see trap 7.

## 3. Duration units differ, and the unit is often not "frame" → C1

Observed units for the same-looking `<action>-<number>` syntax:
pixel frames · **latents** (×4 pixel frames) · fixed-length action blocks · 2-second text
lines · autoregressive chunks.

`w-31` meaning 31 latents = 124 frames, not 31 frames, is a 4× duration error.

## 4. README fps ≠ config fps ≠ writer fps → C2

Observed in the same repo: README says 25 FPS, config constant says `sample_fps = 16`,
and the actual `export_to_video(..., fps=17)` call writes 17. Wall-clock duration is set by
the **writer**; the config value may only affect training-time conditioning.

**Confirmed empirically**: one model whose README documents 24 fps produced benchmark videos
at **19 fps** (1145 frames ≈ 60.3 s). Had the plan trusted the README, the requested duration
would have been wrong by 26%.

Record all values you find, state which governs duration, and **measure the produced file**
rather than trusting any of them.

## 5. Absolute motion scale may be discarded before conditioning → D3

Observed in more than one repo: per-frame translations are divided by their max norm before
being turned into the geometric condition. Consequence: a trajectory at speed 0.05/frame and
one at 0.5/frame produce **identical** conditioning if the shape is the same.

Where this happens, **absolute speed is not controllable** through the geometry path — only
trajectory shape and *relative* speed variation within a clip. If your benchmark claims
matched speeds across models, this must be disclosed. Some repos feed the raw action vector
through a second path, which partially restores magnitude information; check whether both
paths exist.

## 6. First chunk length ≠ later chunk length → C4

Observed: first clip 57 frames, every subsequent clip 40. Total is
`57 + (N-1)*40`, not `N*40`. Segments therefore cannot all be exactly equal; pick a policy
and record it (see `output_spec.md`).

## 7. Two code paths, same-looking control, different semantics → A3

Observed: one repo ships an autoregressive-forcing entry point where the per-segment number
is a frame-count weight, **and** a separate helper where the analogous number is a magnitude
with a fixed 33-frame duration, **and** the two use different hardcoded intrinsics.

Determine which path the launch script actually calls. Never mix constants across paths.

## 8. Repo bugs that only fire in multi-GPU or looped use → F1, F5

Observed:
- a keyword-argument name mismatch in the sequence-parallel attention wrapper — single-GPU
  path never touches it, any multi-GPU run crashes immediately;
- `destroy_process_group()` + `exit()` at the end of the per-sample generate function — a
  batch loop renders exactly one sample and the process exits with no error;
- a pinned dependency incompatible with the installed torch, which only surfaces at import.

Fix in `patches/`, and check the patch is applied (`git diff --stat`) at the start of every
run rather than assuming.

## 9. Intrinsics: classify before you inject → E1–E3

Observed hardcoded values implying FOVs of ~63°, ~79°, ~89° and exactly 90° across different
repos, some normalised and some in pixels, some rescaled internally against a **fixed
reference resolution** rather than the actual working one.

Two symmetric mistakes:

- **Injecting where nothing is wanted.** Only a genuinely camera-conditioned model
  (case 4 in `output_spec.md §4`) consumes `arena_inputs/*_intrinsics/`. A model with no
  camera model, a hardcoded/derived virtual pinhole, or its own runtime estimator needs
  nothing from us — do not manufacture an injection point, and do not patch a hardcoded
  constant unless the user asks.
- **Passing ours through raw.** Our values are in **original-image pixels**
  (`cx ≈ (W-1)/2`). Every model resizes first, and often changes aspect ratio. Scale and crop
  the intrinsics to match, and check whether the repo wants pixels or normalised values, and
  against which resolution.

Either way, record the source FOV and the model's effective FOV. Our data spans **22°–90°**
horizontal, so a fixed-FOV model is mismatched for most samples; that is a disclosable caveat,
not something to silently fix.

## 10. Motion may be eased, not step-constant → D4

Observed: target velocity approached by exponential smoothing with separate press/release
time constants (~0.45 s / ~1.0 s). Even with an exactly correct frame count, the first
fraction of a second of each segment is still accelerating and the previous action is still
coasting.

Consequence for evaluation: segment boundaries are not sharp. Prefer discarding a fixed
transition window at the start of each segment **for all models**, so the eased model is not
penalised relative to step-constant ones.

## 11. "It uses X, so it must not use Y" → G1

A conclusion of the form "this model is action-conditioned, therefore it has no camera
geometry" was asserted and turned out to be **wrong**: the repo integrated the action stream
into camera poses and built a ray-map condition, *in addition* to a separate action module.

Injection mechanisms are not mutually exclusive. One repo used an additive ray-map embedding
**and** a projection-style camera position encoding inside attention. Grep for all of them
before characterising a model, and prefer `UNVERIFIED` over a tidy story.

## 12. Example files are ground truth; README prose is a hint

Where a repo ships an example input, load it and print shapes, dtypes and a few values.
Observed disagreements between README description and shipped file, and one case where the
example's stated image dimensions were only a reference for intrinsics, not the output
resolution.

## 13. Package name ≠ import name; installing one thing un-installs another

Environment hazards seen repeatedly: a drop-in replacement package that provides the same
import name but a different distribution name, so the dependency resolver pulls the original
back in; a headless variant of a library being silently replaced by the GUI variant as a
transitive dependency of an unrelated install.

After any install into a model env, re-check the packages your run depends on rather than
assuming the env is still good.

## 14. Prompts are per (view, domain, image) and not interchangeable → H1

There are four caption files — `{first_view,third_view}/prompt_{real,style}.txt`, 25 rows each.
Every `first_view` caption is prefixed `First-person view.` and every `third_view` one
`Third-person view.`, so the caption itself encodes the camera setup.

Failure modes seen or easily reachable:

- loading one prompt file and reusing it across views → the model is told "third-person" while
  being handed a first-person image;
- off-by-one row indexing → every video gets its neighbour's caption, which is nearly invisible
  in spot checks and poisons any text-alignment metric;
- an adapter "improving" or regenerating captions → the text then carries scene information the
  benchmark never supplied, making that domain incomparable.

Use `{view}/prompt_{domain}.txt` row `i` for image `i`, verbatim, and record
`prompt.source` as `file:row` in the manifest so it is auditable.

Note the zip's own `README.txt` is stale here (it claims only `third_view` has prompts) — see
trap 12.

## 15. Inputs are not uniform in size or aspect ratio → E5, H3

Measured: `real` images span 1024×768 … 1920×1200 (AR 1.33–1.79); `style` is uniformly
1024×1024 (AR 1.00). Target models generate at ~1.7–1.8 AR.

So every sample undergoes a non-trivial resize, and square `style` inputs need a real
crop/pad decision. An adapter that hardcodes one scale factor, or assumes a fixed input size,
will be wrong for most of the suite. Decide one policy per model, apply it uniformly, document
it.
