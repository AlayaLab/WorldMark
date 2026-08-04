# WorldMark — Methods, Rationale & Formulas

9 metrics in 3 categories. `FPS = 16`. Action protocol (`protocol.py`): 20 s per
key — `W` forward / `S` back / `A` strafe-left / `D` strafe-right / `L` yaw-left /
`R` yaw-right.

**Three design principles**
1. **Motion-indexed, not time-indexed.** Sampling and pairing are done over
   cumulative optical-flow arc length, not wall-clock. A "fast" vs "slow" execution
   of the same action would otherwise traverse different amounts of motion and be
   incomparable.
2. **Deterministic & reproducible.** Every metric is bit-reproducible. Q-Align and
   VGGT-Omega are verified bit-identical run-to-run; SLAM back-ends (DROID-SLAM,
   ViPE) are **not used** for Global Memory because they are non-deterministic
   (measured ~0.1–0.3 % pose drift run-to-run, `array_equal = False`).
3. **Perceive once, score many.** The expensive perception (SEA-RAFT flow + DA3
   depth) is cached; changing a metric formula only re-scores the cache.

**Final reporting.** In the master table all 12 columns are normalized to
**[0, 100], higher is better**, 2 decimals. Direction Accuracy / Direction Purity /
Motion Stability / Response Latency are split into **trans** (pool of W/S/A/D
segments) and **rot** (L/R segments); Revisit Memory pools translation + rotation
round-trips into one number. Coloring: green = column best, yellow = 2nd, red = worst.
By default Open-Oasis (uniformly failing) and Genie3 (only 5 videos) are excluded.

---

## 1. Action Dynamics (M1–M4)

Shared perception (`flow_eval.py`): SEA-RAFT per-frame global flow giving `Sx`
(horizontal) and `Srad` (radial about the focus of expansion). Expected-flow
templates: `W` radial-expand (Srad+), `S` contract (Srad−); `A`/`L` content
flows +x (Sx+), `D`/`R` −x. Depth-Anything-3 depth on lat/yaw segments fits
`u = a·(1/Z) + b`, separating **translation parallax** (∝ a) from **rotation**
(global shift ∝ b) — needed because strafing and turning share the same horizontal
flow template.

**Direction Accuracy** — is the induced motion in the commanded direction?
```
v   = sign(k) · signed_channel(k)[segment]
dir = mean(v) / (mean(|signed_channel|) + 1e-9)        ∈ [-1, 1]
w   = p_trans (lat) | p_yaw (yaw) | 1 (fwd)            # depth-parallax motion-type likelihood
dir_gated = dir · w                                   # "right direction AND right motion type"
```
Rejected alternative: DA3 pose axis-angle version — depends on monocular pose scale
and is flip-sensitive; the flow version is scale-free and flip-immune (DA3 kept only
for the type weight).

**Direction Purity** — is the motion confined to the commanded axis (no leakage)?
```
lateral = p_trans·|Sx|,  yaw = p_yaw·|Sx|,  fwd = |Srad|   # W/S: lateral=yaw=0.5|Sx|
purity  = on_axis / (lateral + yaw + fwd + 1e-9)          ∈ [0, 1]
```

**Motion Stability** — once responding, is the speed steady (no hiccups / decay)?
Baseline `g = P70(v)` (normal speed, not dragged down by failures):
```
sustain = mean(smoothed_v ≥ 0.4g)                     # fraction of time at normal speed
hiccups = hysteresis count of dips below 0.4g (≥0.3s) that recover above 0.5g; hr = hiccups / seconds
decay   = clip(mean(v last 3s) / mean(v first 3s), 0, 1.2)
score   = sustain · exp(−hr/0.5) · min(1, decay/0.7)  ∈ [0, 1]
```
Rejected: a spectral-arc-length "reversal smoothness" metric — not robust on mean
flow and overlapped with the others; removed.

**Response Latency** — how quickly the model responds after a command change.
Uses an **absolute** reference (a self-referential threshold lets weak responders
pass): `v_ref = P75(|channel|)`, threshold `0.5·v_ref`. Gate: if segment steady-state
`≤ 0.1·v_ref` → no response, score 0.
```
latency = first time smoothed_v ≥ 0.5·v_ref after the command (else = segment length L)
score   = 1 − min(latency, L) / L                     ∈ [0, 1]     # reported ×100
```

---

## 2. World Memory (C1–C3)

Motion-indexed sampling; **C1/C2 use no 3D reconstruction** (reconstruction would
collapse first on exactly the degraded content we test — a circular dependency;
geometry is handled by C3). Appearance backbone: **DINOv2-base patch-mean (pmean)**
— mean of patch tokens, L2-normalized, distance `1 − cos`. More sensitive to local
texture change than the CLS token, more robust to viewpoint residual than per-patch
or LPIPS.

**Local Memory** (adjacent, short timescale): no abrupt frame-to-frame mutations.
```
sample: one frame per δ of cumulative flow (δ = median 0.5 s flow); <5 frames -> frozen -> not evaluable
hard-cut rate: TransNetV2 (frame-level) scene count > 1
mutation rate: fraction of adjacent pmean-distances > TAU_MUT (0.10);
               exclude frame pairs within ±0.5 s of a hard cut (avoid double counting)
Local Memory = 100 · (1 − hardcut_rate) · (1 − mutation_rate)   # non-frozen videos only
```
Rejected: z-score spikes (miss "uniformly bad" — no outliers); CLS (blunt to repaint);
delegating to Q-Align (kept deterministic + orthogonal, measured Spearman ≈ 0.35 vs
quality). Frozen videos are excluded (penalized by Action Dynamics instead), closing a
"freeze + occasional jump" loophole.

**Revisit Memory** (leave a pose, return to it): is the world remembered?
```
gates (all self-derived from flow): direction, response onset, gain symmetry, (rot only) turn magnitude ≥ 15°
pair by equal cumulative arc length: 10 depths uniform in [reach_lo, min(out, return arc)],
    reach_lo raised so paired frames are ≥ MIN_GAP_S (2 s) apart; absolute tolerance (trans 30 px / rot 3°)
    motion index: trans = |flow|, rot = |Sx| (yaw proxy; focal length cancels between out/return)
per-pair pmean distance d_k;  Revisit = 100 / (1 + mean(d_k)/TAU_C),  TAU_C = 1.0
```
Rejected: VGGT co-visible-point pairing (circular dependency); equal-time pairing
(sensitive to gain asymmetry). trans and rot share TAU_C so their difference is
meaningful — a large trans−rot gap is a "short-context" signature (remembers while
translating, forgets once content leaves the frame under rotation).

**Global Memory** (whole-trajectory geometry): globally consistent camera path +
point cloud, via **VGGT-Omega** feed-forward joint reconstruction (1 fps, ≤64 frames):
one forward pass yields per-frame pose + depth + world points; cross-frame geometry
error `e = global_median_e` (lower is raw-better). Reported normalized:
`100 / (1 + e/τ_G)`, `τ_G = 0.05`. Rejected: DROID-SLAM / ViPE (non-deterministic);
naive per-frame VGGT photometric term (too local to be "global").

---

## 3. Visual Quality

**Q-Align / OneAlign** (mPLUG-Owl2 multimodal LLM), video mode at 1 fps. Reads the
logits of five rating tokens {excellent, good, fair, poor, bad}, softmax-weighted to
{5,4,3,2,1}. No sampling → deterministic. `Perceptual Quality` = quality task (VQA);
`Aesthetic Quality` = aesthetics task (IAA). Reported normalized `100·(x−1)/4`.
Rejected: VBench (MUSIQ + CLIP-aesthetic) and VQ-Insight — Q-Align is a single
stronger, deterministic model.

---

## Constants (frozen; C1/C2 thresholds pending calibration)

| Const | Value | Use |
|---|---|---|
| `DELTA_FRAC` | 0.5 | Local Memory equal-motion step (0.5 s flow) |
| `NMIN` | 5 | min sampled frames else frozen |
| `TAU_MUT` | 0.10 | Local Memory mutation threshold (initialized from set P90, frozen) |
| `CUT_GUARD` | 8 frames | exclude ±0.5 s around hard cuts |
| `N_PAIRS_C2` | 10 | Revisit pairs per reversal |
| `MIN_GAP_S` | 2.0 | min temporal gap of a revisit pair |
| `TAU_C` | 1.0 | Revisit score scale (shared trans/rot) |
| `TOL_ARC` / `ROT_TOL_DEG` | 30 px / 3° | Revisit arc-match tolerance |
| `FOV_DEG` | 70 | rot angle↔flow conversion (not used in pairing) |
| `ROT_MAG_GATE_DEG` | 15 | Revisit rotation magnitude gate |
| `TAU_G` | 0.05 | Global Memory normalization scale |
| `DIR_GATE`/`GAIN_GATE`/`COV_GATE` | 0.3/0.5/0.3 | Revisit gates |
