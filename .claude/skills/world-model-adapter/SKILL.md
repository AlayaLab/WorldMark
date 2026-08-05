---
name: world-model-adapter
description: Adapt a new world-model / camera-controlled video-generation repo to the WorldMark arena inputs, and batch-render the videos. Use when the user points at an inference repo (Matrix-Game, LingBot, GameCraft, Sana-WM, Lyra, YUME, AlayaWorld, DreamX, WorldPlay, or any new one) and wants to generate the benchmark videos, or asks how to convert the benchmark's WSADLR action protocol into that model's native control format.
---

# World-model adapter

Take `arena_inputs/` (starting images + per-image intrinsics + action assignments) and produce
one mp4 per (image × action) in that model's native control format.

The investigation is yours to figure out. The **I/O contract** and the **verification gates**
are not negotiable.

## Required from the user — exactly two things

1. **The model repo** — a local path or a URL to clone. Everything about the control format is
   derived from it by reading the source; nothing is assumed from the model's name.
2. **Which view and which domain** — the unit of work is one **(view, domain)** pair
   = 125 videos. `view` ∈ `first_view` / `third_view`, `domain` ∈ `real` / `style`.

**If either is missing, ask before doing anything else.** Do not start probing without the
repo, and do not start G4 without knowing the scope.

Interpreting a partial answer:

| They said | Render |
|---|---|
| a specific pair ("first-person real", "第一视角 real") | that pair only |
| a view only ("first_view") | `real` + `style` for that view |
| a domain only ("style") | both views for that domain |
| "all" / "全部" | all four = **500 videos**, hours of multi-GPU — say the cost, then confirm |

Gates G0–G3 only ever need **one** pair. Run them on the user's first-choice pair and report
before expanding.

## Language

Converse in the user's language. Artifacts follow fixed rules regardless:

- `manifest.jsonl` field **names** and enum values: English, exactly as in `output_spec.md §3`
  — this is a machine contract, never localise the keys.
- Filenames, paths, `model_id`, `MODEL` tags: ASCII, as specified.
- `PROBE.md` / `README.md` prose: the user's language is fine, but keep every `file:line`
  citation, identifier, and code snippet verbatim from the source.

## Hard rules

1. **Every claim about the repo needs `file:line` evidence.** Never infer behaviour from a
   variable name, a README sentence, or "how other repos do it". Read the consumer of the
   value. Past defects came almost entirely from plausible-looking assumptions.
2. **Do not guess key semantics.** `a`/`d` mean yaw in some repos and strafe in others. Find
   and quote the mapping table.
3. **No GPU until gates G0–G1 pass.** A full run is hours of multi-GPU time.
4. **Never edit `arena_inputs/`.** Repo edits go in `patches/` as patch files, and may change
   only *how control input gets in* — never the conditioning semantics, magnitudes, or
   constants (see `output_spec.md §6`). Changing those needs the user's approval.
5. **Run `generation/check_delivery.py` yourself and paste its output before reporting done.**
   A delivery that does not exit 0 is not a delivery. Never report success on the strength of
   your own bookkeeping — the gate decodes the videos, you do not get to vouch for them.
6. If something cannot be determined from source, write `UNVERIFIED` in `PROBE.md` and tell
   the user. Do not fill the gap with a guess.

## Workflow

### 1. Read the inputs first

`arena_inputs/action_protocol.txt` is authoritative for the action semantics and
`arena_inputs/README.txt` describes the layout. Still **list the directories yourself** to
confirm — the inputs can be revised, and a doc that has drifted is invisible until it costs you
a full run.

**[references/output_spec.md](references/output_spec.md) §1** records the layout and the
verified facts about the data — including three things that catch people out:

- intrinsics are **per-image, in original-image pixels** — but only a genuinely
  camera-conditioned model consumes them; classify the model into one of the four cases in §4
  first (real FOV spans 22°–90°, so a hardcoded-FOV model's mismatch is worth recording);
- all four `prompt_*.txt` exist and **differ by view** (captions are prefixed
  `First-person view.` / `Third-person view.`) — never cross views or domains (§5);
- image aspect ratios are not uniform (1.00 for `style`, 1.33–1.79 for `real`).

### 2. Probe

Work through **[references/probe_checklist.md](references/probe_checklist.md)** and write
`adapters/<model_id>/PROBE.md`. Every item gets evidence or `UNVERIFIED`.

Opening sweep (adjust to the repo):

```bash
# control surface
grep -rn "add_argument" --include="*.py" . | grep -iE "action|camera|pose|traj|motion|control|prompt"
# native primitives
grep -rln "plucker\|Plucker\|ray_condition\|rays_d\|prope\|PRoPE" --include="*.py" .
grep -rn "c2w\|w2c\|intrinsic\|extrinsic" --include="*.py" . | head -30
grep -rniE "keyboard|mouse|wasd|action_dict|KEY_TO|VALUE_MAP" --include="*.py" .
# time / scale constants
grep -rniE "fps|move_speed|rotate_speed|_SPEED|OFFSET|SENSITIVITY|duration" --include="*.py" .
# batch blockers
grep -rn "sys.exit\|exit()\|destroy_process_group" --include="*.py" .
```

Read **[references/known_traps.md](references/known_traps.md)** before concluding. Every entry
is a mistake that has actually been made; each maps to a checklist item.

### 3. Build

Produce the deliverables in **output_spec §6**: `PROBE.md`, `adapt.py` (CPU-only),
`run_batch.py` (resumable), `test_golden.py`, `patches/`, `README.md`.

First decide the **integration level** (output_spec §6) — many repos cannot be driven purely
from outside:

- **A** a batchable file/CLI control surface already exists → no repo changes;
- **B** control input is interactive-only or otherwise unbatchable → patch in a file/flag path
  that feeds the *same* internal call;
- **C** per-sample inference is not separable from loading → patch, and have `run_batch.py`
  import the pipeline class directly instead of shelling out to the repo's CLI.

B and C are normal. State the level in `PROBE.md` and `README.md`.

### 4. Verify — gates in order

| Gate | Cost | Catches |
|------|------|---------|
| **G0** `test_golden.py` | none | mapping / timing / intrinsics-math regressions |
| **G1** `adapt.py --dry-run` + trajectory assertions | none | sign errors, wrong axis, wrong units |
| **G2** 1 sample, shortest action (id 1–5) | minutes | crashes, env, arg errors, length mismatch |
| **G3** watch that 1 video | minutes | **sign errors that pass every numeric test** |
| **G4** full 125 per (view, domain), resumable | hours | — (scope: see top of file) |
| **G5** `generation/check_delivery.py` | seconds | naming, missing/extra files, undecodable video, out-of-tolerance duration |

G3 is not optional. A flipped yaw sign satisfies G0–G2 and yields a video turning the wrong
way. Watch it, or have the user watch it, before G4.

For G1 assert properties, not eyeballed numbers: during `W` the camera position advances
monotonically along the forward axis; during `R` accumulated yaw is monotone and translation
~0; if the world is y-up the y component stays ~0. Record sampled poses in `PROBE.md`.

**G5 is the acceptance gate and it decides.** It decodes the delivered mp4s and ignores any
manifest, so it cannot be talked around. Run it after G2 on the single sample, and again after
G4 on the full set; a delivery that does not exit 0 is not delivered.

```bash
python generation/check_delivery.py \
    --arena arena_inputs/ --delivery real/first/<MODEL> \
    --view first_view --domain real
```

If a duration genuinely cannot land inside the default ±10 % because of the model's rollout
granularity, do **not** silently widen `--tolerance`: report the measured deviation to the user
and let them decide the tolerance for that model.

### 5. Report

Concisely tell the user:
- native format, and the exact string/file the model receives for one example action
- achieved resolution / fps / frames / seconds vs. the 20 s-per-segment target, with deviation
- which of the four intrinsics cases (§4) this model is, and any FOV mismatch
- prompt handling (used verbatim / truncated / ignored)
- anything `UNVERIFIED`
- anything that makes cross-model comparison unsafe (traps 5, 10, 14)

## Conventions

- One `model_id` = one adapter. Two code paths in the same repo with different control
  semantics (this happens) are two adapters, or one adapter with an explicit documented switch.
- Resolution and fps are **deliberately not unified** across models — evaluation resizes and
  time-normalises. Your job is to record the actuals, not to match a target.
- Long runs: progress lives in `jobs.jsonl` on disk, never in conversation context.
