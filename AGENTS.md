# WorldMark benchmark — agent instructions

This repo holds the WorldMark arena inputs and per-model adapters for camera-controlled
video-generation ("world model") repos.

## Adapting a model repo and rendering the benchmark videos

When asked to connect a model repo to the benchmark, batch-generate the videos, or convert the
`W S A D L R` action protocol into a model's native control format, **follow the skill in
`.claude/skills/world-model-adapter/`**. It is plain Markdown and applies to any agent:

| File | Read it for |
|------|-------------|
| `.claude/skills/world-model-adapter/SKILL.md` | the workflow, hard rules, and verification gates G0–G5 — **start here** |
| `.claude/skills/world-model-adapter/references/output_spec.md` | the I/O contract: input layout, mp4 naming/directories, intrinsics cases, prompt rules, integration levels A/B/C, patch discipline |
| `.claude/skills/world-model-adapter/references/probe_checklist.md` | the questions to answer about the repo, each needing `file:line` evidence |
| `.claude/skills/world-model-adapter/references/known_traps.md` | 15 mistakes that have actually been made; read before concluding anything |
| `generation/check_delivery.py` | the acceptance gate — run it, do not vouch for a delivery yourself |

The I/O contract and the gates are **not optional**. The investigation method is yours to
choose.

### Two things you need from the user before starting

1. **The model repo** — local path or URL.
2. **Which view and which domain** — one `(view, domain)` pair = 125 videos.
   `view` ∈ `first_view` / `third_view`, `domain` ∈ `real` / `style`; all four = 500 videos.

Ask if either is missing.

### Non-negotiables, in short

- Every claim about a repo needs `file:line` evidence. Never infer control semantics from a
  variable name, a README sentence, or how a different repo does it.
- Never edit `arena_inputs/`. Repo patches live in `adapters/<model_id>/patches/` and may change
  only *how control input gets in* — never magnitudes, constants, or the conditioning path.
- No GPU work until the free gates (G0, G1) pass; watch one video (G3) before a full run.
- Run `generation/check_delivery.py` and paste its output before reporting done.

## Repo layout

```
arena_inputs/                  # benchmark inputs (read-only): images, actions, prompts, intrinsics
generation/check_delivery.py   # the acceptance gate
adapters/<model_id>/           # you create these — one per model, per output_spec.md §6
evaluation/                    # scoring: run_eval.sh + worldmark/ (9 metrics)
.claude/skills/                # the skill above
```

Once a delivery passes the gate, scoring it is a separate step — see `evaluation/README.md`.
Do not modify anything under `evaluation/`; producing videos is the whole job here.
