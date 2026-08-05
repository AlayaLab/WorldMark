# Generation

Turn **[`../arena_inputs/`](../arena_inputs/)** into one mp4 per `(image × action)` in your
model's own control format.

```
{domain}/{view}/{MODEL}/{image:03d}_{action:03d}.mp4     e.g. real/first/MYMODEL/016_014.mp4
```

One `(view, domain)` pair = 25 images × 5 actions = **125 videos**; all four = **500**.

## The contract

**[`../.claude/skills/world-model-adapter/references/output_spec.md`](../.claude/skills/world-model-adapter/references/output_spec.md)**
is authoritative: input layout, filenames and directories, the four intrinsics cases, prompt
rules, and integration levels A/B/C for repos that cannot be driven from outside.

Two things people get wrong often enough to repeat here:

- **Keep your native resolution and fps.** Evaluation resizes and time-normalises; there is no
  target to match. Measure and report what you actually produced.
- **`a` / `d` mean yaw in some repos and strafe in others.** Read the consumer of the value in
  the model's source rather than trusting the name.

`../.claude/skills/world-model-adapter/references/known_traps.md` lists 15 mistakes that have
actually been made — worth a skim before you conclude anything about a repo.

## Using an agent

The skill in `../.claude/skills/world-model-adapter/` does the adapter and the batch render.
It needs the **model repo** and **which view + domain**; see the repo
[README](../README.md#1%EF%B8%8F%E2%83%A3-generate-videos).

## `check_delivery.py`

The acceptance gate. It **decodes the videos** and ignores any manifest, so it is the arbiter
of whether a delivery is scoreable:

```bash
python check_delivery.py \
    --arena ../arena_inputs/ --delivery /path/to/real/first/MYMODEL \
    --view first_view --domain real
```

Checks filenames, exact completeness against `{view}/{domain}_action.txt`, decodability,
per-clip duration (20 s per segment, default ±10 %), and resolution/fps consistency across the
delivery. **Exit 0 = ready to score.**

Options:

| Flag | Default | Use |
|---|---|---|
| `--tolerance` | `10` (%) | widen only if the model's rollout granularity genuinely cannot land inside it — and say so when you report |
| `--probe-backend` | `auto` | `ffprobe` or `cv2`; `auto` tries ffprobe then falls back |
| `--sec-per-segment` | `20` | leave alone unless you know why |
| `--json-out` | — | machine-readable report |

Requires `ffprobe` on `PATH`, or `opencv-python` for the `cv2` backend.

Once it exits 0, score the delivery: **[`../evaluation/README.md`](../evaluation/README.md)**.
