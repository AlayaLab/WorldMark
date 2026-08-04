"""C1 (Local Memory / adjacent mutation) + C2 (Revisit Memory), final version.

C2: back-and-forth motions (actions 6/8/11/14). Equal-arc-length pairing with a
    latency-aware return start, gated by (M1 direction / M3 response / gain
    symmetry / coverage), scored with LPIPS (low level) + DINOv2 (mid-level
    semantics) plus a decay curve. An equal-time pairing baseline is also emitted
    for comparison.
C1: on equal-motion sampled frame pairs, look for LPIPS/DINOv2 distance spikes
    (z-score) plus TransNetV2 hard cuts.
Appearance metrics: LPIPS + DINOv2. (DreamSim is dropped because it conflicts
    with WorldScore's transformers 4.37; DINOv2 carries the mid-level semantic role.)

Usage: SHARD/NSHARD CUDA_VISIBLE_DEVICES=X python consistency_v2.py
"""
import os, sys, glob, json, numpy as np
import wm_config as config
import torch, torch.nn.functional as F
from decord import VideoReader, cpu
import protocol as P

RF = config.flow_dir()
CACHE = config.flow_cache()
BASE = config.VIDEO_ROOT
OUT = config.consistency_dir()
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
FPS = 16.0
REVISIT = {6, 8, 11, 14}          # opposing-translation round trips
YAW_REV = {7, 12}                 # opposing-rotation round trips (LR / LRL)
DELTA_FRAC = 0.5                  # C1 equal-motion step baseline (median flow over 0.5s)
Z0 = 2.5                          # mutation spike z threshold
TAU_MUT = 0.10                    # C1 absolute mutation threshold: an adjacent pmean distance > this counts as
                                  # one mutation event (purely temporal, catches "uniformly bad").
                                  # Origin = P90 of pmean distances on the initial reference set; now FROZEN and
                                  # not recomputed per set (keeps history comparable). To be finally calibrated
                                  # against a UE-injection experiment (precision/recall on known mutation frames).
                                  # Do NOT change this to "take P90 on every run".
N_PAIRS_C2 = 10                   # frame pairs sampled per reversal point (uniform in arc-length space, fixed)
MIN_GAP_S = 2.0                   # minimum time gap between paired frames (s): too close = near-identical, no info
TAU_C = 1.0                       # C2 score scale (fixed): C2 = 100/(1 + d_bar/TAU_C)
TOL_ARC = 30.0                    # (translation) equal-arc-length match absolute tolerance (cumulative flow px):
                                  # >> single-frame step, << typical arc length
# --- Rotation round trips (C2_rot, scheme A): cumulative signed lateral flow Sx as a yaw proxy; pairing is
#     independent of FOV (f cancels on both sides). The two absolute-angle gates below are converted to flow
#     using a nominal horizontal FOV: 1 degree ~= W/FOV_DEG px cumulative flow.
#     A 20% error in the FOV assumption only shifts gate tightness by 20%; it does not affect pairing correctness.
FOV_DEG = 70.0                    # nominal horizontal FOV (fixed); only for angle<->flow conversion, not for pairing
ROT_MAG_GATE_DEG = 15.0           # magnitude gate: outbound total rotation < 15deg -> view never left, no revisit, skip
ROT_TOL_DEG = 3.0                 # rotation pairing absolute angle tolerance (converted to flow, passed to c2_pairs_arc)
ROT_LO_DEG = 5.0                  # outbound pairing lower bound: frames < 5deg from the reversal are near-identical, skip
DIR_GATE = 0.3; GAIN_GATE = 0.5; COV_GATE = 0.3; NMIN = 5
C2FEAT = f"{OUT}/c2_feat"; os.makedirs(C2FEAT, exist_ok=True)   # cls/ppatch/lpips silent archive

# ---------------- appearance models ----------------
import lpips as lpips_lib
_lpips = lpips_lib.LPIPS(net="alex").to(dev).eval()
from transformers import AutoImageProcessor, AutoModel
_dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
_transnet = None
def transnet():
    global _transnet
    if _transnet is None:
        from transnetv2_pytorch import TransNetV2
        _transnet = TransNetV2().to(dev).eval()
    return _transnet

def _t(img):  # HWC uint8 -> [1,3,H,W] float [0,1]
    return torch.from_numpy(img).float().permute(2, 0, 1)[None].to(dev) / 255.0

@torch.no_grad()
def lpips_dist(a, b):  # a,b HWC uint8
    ta = F.interpolate(_t(a), (224, 224), mode="bilinear", align_corners=False) * 2 - 1
    tb = F.interpolate(_t(b), (224, 224), mode="bilinear", align_corners=False) * 2 - 1
    return float(_lpips(ta, tb).item())

@torch.no_grad()
def dino_feats(frames):  # list HWC uint8 -> [N,768] normalized CLS
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames]
    out = []
    for i in range(0, len(pil), 16):
        inp = _dino_proc(images=pil[i:i+16], return_tensors="pt").to(dev)
        out.append(F.normalize(_dino(**inp).last_hidden_state[:, 0], dim=-1))
    return torch.cat(out, 0)

@torch.no_grad()
def dino_feats_full(frames):  # -> cls [N,768] norm, patch [N,P,768] norm
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames]
    cls = []; pat = []
    for i in range(0, len(pil), 16):
        inp = _dino_proc(images=pil[i:i+16], return_tensors="pt").to(dev)
        h = _dino(**inp).last_hidden_state            # [b,1+P,768]
        cls.append(F.normalize(h[:, 0], dim=-1))
        pat.append(F.normalize(h[:, 1:], dim=-1))     # per-patch normalization
    return torch.cat(cls, 0), torch.cat(pat, 0)

def zspikes(d):
    d = np.asarray(d)
    if len(d) < 3 or d.std() < 1e-9: return 0
    return int((((d - d.mean()) / (d.std() + 1e-9)) > Z0).sum())


# ---------------- load video signals (only depends on the flow cache, not the M1-M4 JSON; self-contained) ----------------
def load(model, stem):
    z = np.load(f"{CACHE}/{model}__{stem}.npz", allow_pickle=True)
    Sx, Srad = z["Sx"], z["Srad"]; aid = int(z["action"])
    mag = np.sqrt(Sx**2 + Srad**2)
    sig = Srad if P.keys_of(aid)[0] in ("W", "S") else Sx    # signed (direction)
    return dict(Sx=Sx, Srad=Srad, mag=mag, sig=sig, aid=aid, keys=P.keys_of(aid))

# expected flow sign (protocol section 2.5): W radial expansion +, S contraction - ; A/L content flows right +x, D/R -x
FAM_SIGN = {"W": +1, "S": -1, "A": +1, "D": -1, "L": +1, "R": -1}
def _chan(Sx, Srad, k):                       # the signed channel for this key
    return Srad if k in ("W", "S") else Sx
def seg_dir(Sx, Srad, keys, i, N, n):
    """Direction score of segment i computed from flow = expected sign * net flow / total flow in [-1,1]
    (equivalent to the M1 dir core)."""
    k = keys[i]; lo = int(round(i/n*N)); hi = int(round((i+1)/n*N))
    ch = _chan(Sx, Srad, k)[lo:hi]; tot = float(np.sum(np.abs(ch)))
    if tot < 1e-6: return 0.0
    return FAM_SIGN[k] * float(np.sum(ch)) / tot
def return_start(Sx, Srad, keys, b, t0, seg_hi, frac=0.2, win=3):
    """Detect the return start from flow: the first frame where the rolling mean of the return-direction
    flow after the reversal >= frac * return steady-state speed. None if never responded."""
    k = keys[b]; ch = _chan(Sx, Srad, k)[t0:seg_hi]
    if len(ch) < win + 1: return None
    ret = FAM_SIGN[k] * ch; sr = float(np.mean(np.abs(ch)))
    if sr < 1e-6: return None
    thr = frac * sr
    for j in range(len(ret) - win):
        if float(np.mean(ret[j:j+win])) >= thr: return t0 + j
    return None


# ---------------- C1 ----------------
def c1_process(model, stem, delta, vr):
    d = load(model, stem); mag = d["mag"]; N = len(mag); NF = len(vr)
    cum = np.cumsum(mag); idx = []; nxt = delta
    for i, c in enumerate(cum):
        if c >= nxt: idx.append(i); nxt += delta
    out = {"n_pairs": max(0, len(idx) - 1)}
    if len(idx) < NMIN:
        out["na"] = "frozen"; return out
    fr = vr.get_batch([min(i, NF - 1) for i in idx]).asnumpy()
    _, pat = dino_feats_full(fr)                          # patch tokens [N,P,768]
    pm = F.normalize(pat.mean(1), dim=-1)                 # patch-mean normalized [N,768] (same as C2 main metric)
    pm_d = np.array([1 - float(F.cosine_similarity(pm[i:i+1], pm[i+1:i+2]).item()) for i in range(len(idx)-1)])
    lp_d = np.array([lpips_dist(fr[i], fr[i+1]) for i in range(len(idx)-1)])
    out.update(pmean_level=float(np.mean(pm_d)), pmean_spikes=zspikes(pm_d),
               mut_rate=float(np.mean(pm_d > TAU_MUT)),        # absolute-threshold mutation event rate (main temporal component)
               mut_events=int(np.sum(pm_d > TAU_MUT)),          # raw mutation count (for reporting; the rate is more readable)
               pmean_dists=[round(float(x), 4) for x in pm_d],  # full sequence stored so the threshold can be retuned later
               idx=[int(i) for i in idx],                       # sampled frame indices (to exclude pairs near hard cuts)
               lpips_level=float(np.mean(lp_d)), lpips_spikes=zspikes(lp_d))
    return out

@torch.no_grad()
def c1_hardcut(vr):
    """No ffmpeg dependency: decord reads frames -> 48x27 -> sliding-window TransNetV2 -> scene count."""
    try:
        import cv2
        m = transnet(); NF = len(vr)
        fr = vr.get_batch(list(range(NF))).asnumpy()
        small = np.stack([cv2.resize(f, (48, 27)) for f in fr]).astype(np.uint8)   # [T,27,48,3]
        pad = np.concatenate([np.repeat(small[:1], 25, 0), small, np.repeat(small[-1:], 25, 0)], 0)
        preds = []
        for st in range(0, NF, 50):
            win = pad[st:st+100]
            if len(win) < 100:
                win = np.concatenate([win, np.repeat(win[-1:], 100-len(win), 0)], 0)
            inp = torch.from_numpy(win)[None].to(dev)          # [1,100,27,48,3]
            sf, _ = m(inp)
            preds.append(torch.sigmoid(sf)[0, 25:75, 0].cpu().numpy())
        p = np.concatenate(preds)[:NF]
        sc = m.predictions_to_scenes(p, 0.5) if hasattr(m, "predictions_to_scenes") else \
             __import__("transnetv2_pytorch").transnetv2_pytorch.TransNetV2.predictions_to_scenes(p, 0.5)
        n = len(sc)
        cuts = [int(s[0]) for s in sc[1:]] if n > 1 else []      # cut points = start of non-first scenes
        return {"hard_cut": n > 1, "n_scenes": int(n), "cut_frames": cuts}
    except Exception as e:
        return {"hard_cut": None, "err": str(e)[:100]}


# ---------------- C2 ----------------
def reversals(aid, N):
    keys = P.keys_of(aid); n = len(keys); out = []
    for b in range(1, n):
        if keys[b] != keys[b-1]:   # direction reversal boundary
            out.append((b, int(round(b/n*N))))
    return out, n

def c2_pairs_arc(mag, t0, seg_lo, seg_hi, t_resp, npairs, tol=TOL_ARC, lo=0.0):
    """Equal-arc-length pairing: sample npairs depths a uniformly in ARC-LENGTH space; for each, find the
    outbound/return frames whose cumulative arc length is closest to a and pair them.
    Depth range = [max(lo, shallowest arc after the return response), min(outbound total arc, return total arc)]
    -- covers the whole memory curve shallow->deep instead of crowding at the deepest end. Absolute tolerance tol:
    if either outbound/return match residual > tol, drop it (excludes unreachable depths).
    Returns pairs=[(i,j,resid)], tot_out, tot_ret; resid=|ret_cum[j]-out_arc[i]| is the two-frame arc mismatch
    (converted to angle for rot diagnostics)."""
    if t0 - seg_lo < 3 or seg_hi - t0 < 3: return [], 0.0, 0.0
    out_arc = np.cumsum(mag[seg_lo:t0][::-1])[::-1]        # out_arc[k] = arc length from seg_lo+k to t0
    ret_cum = np.cumsum(mag[t0:seg_hi])                    # ret_cum[k] = arc length from t0 to t0+k
    tot_out = float(out_arc[0]) if len(out_arc) else 0.0
    tot_ret = float(ret_cum[-1]) if len(ret_cum) else 0.0
    if len(ret_cum) == 0: return [], tot_out, tot_ret
    rstart = max(0, min(t_resp - t0, len(ret_cum) - 1))    # return start = after the response latency
    reach_lo = max(lo, float(ret_cum[rstart]))             # shallowest pairable depth (after response + lo bound)
    reach_hi = min(tot_out, float(ret_cum[-1]))            # deepest pairable depth (reachable on both out and return)
    if reach_hi <= reach_lo: return [], tot_out, tot_ret
    def frames_at(a):
        oi = int(np.argmin(np.abs(out_arc - a))); jj = rstart + int(np.argmin(np.abs(ret_cum[rstart:] - a)))
        return seg_lo + oi, t0 + jj, oi, jj
    # time gap increases monotonically with arc depth: raise the lower end to where gap >= MIN_GAP (dense scan for first hit)
    min_gap_f = MIN_GAP_S * FPS; lo_t = None
    for a in np.linspace(reach_lo, reach_hi, 200):
        i, j, _, _ = frames_at(a)
        if (j - i) >= min_gap_f: lo_t = a; break
    if lo_t is None: return [], tot_out, tot_ret           # whole segment too fast, no pair with gap >= MIN_GAP
    reach_lo = lo_t
    pairs = []; seen = set()
    for a in np.linspace(reach_lo, reach_hi, npairs):      # npairs depths uniform in arc-length space
        i, j, oi, jj = frames_at(a)
        if abs(float(out_arc[oi]) - a) > tol or abs(float(ret_cum[jj]) - a) > tol: continue
        resid = abs(float(ret_cum[jj]) - float(out_arc[oi]))
        if j > i + 1 and (i, j) not in seen:
            seen.add((i, j)); pairs.append((int(i), int(j), float(resid)))
    return pairs, tot_out, tot_ret

def c2_process(model, stem, vr):
    d = load(model, stem); aid = d["aid"]
    if aid in REVISIT: mode = "trans"
    elif aid in YAW_REV: mode = "rot"
    else: return {"na_scope": True}
    sig = d["sig"]; N = len(d["mag"]); NF = len(vr)
    if mode == "rot":
        arc = np.abs(d["Sx"])                       # yaw proxy: cumulative lateral flow magnitude
        W = int(vr[0].shape[1]); deg2arc = W / FOV_DEG   # 1 degree ~= W/FOV px cumulative flow
        tol = ROT_TOL_DEG * deg2arc; lo = ROT_LO_DEG * deg2arc; mag_gate = ROT_MAG_GATE_DEG * deg2arc
    else:
        arc = d["mag"]; deg2arc = None; tol = TOL_ARC; lo = 0.0; mag_gate = 0.0
    Sx, Srad, keys = d["Sx"], d["Srad"], d["keys"]
    revs, n = reversals(aid, N)
    all_pairs = []; ne_reasons = []
    for b, t0 in revs:
        seg_lo = int(round((b-1)/n*N)); seg_hi = int(round((b+1)/n*N))
        # gate 1: direction of the two adjacent segments (computed from flow, not read from M1 JSON)
        if min(seg_dir(Sx, Srad, keys, b-1, N, n), seg_dir(Sx, Srad, keys, b, N, n)) < DIR_GATE:
            ne_reasons.append("dir"); continue
        # gate 2 / return start: detect the return-direction flow start from flow (not read from M3 JSON)
        t_resp = return_start(Sx, Srad, keys, b, t0, seg_hi)
        if t_resp is None: ne_reasons.append("no_resp"); continue
        # gate 3: gain symmetry (outbound/return steady-state speeds; a ratio, so FOV/f cancel)
        so = np.mean(np.abs(sig[seg_lo:t0])); sr = np.mean(np.abs(sig[t0:seg_hi]))
        if min(so, sr)/max(so, sr, 1e-6) < GAIN_GATE: ne_reasons.append("gain"); continue
        # gate 4 (rot only): magnitude gate, outbound total rotation < 15deg -> view never left
        if mode == "rot" and float(np.sum(arc[seg_lo:t0])) < mag_gate: ne_reasons.append("undershoot"); continue
        pa, tot_out, tot_ret = c2_pairs_arc(arc, t0, seg_lo, seg_hi, t_resp, N_PAIRS_C2, tol=tol, lo=lo)
        partial = tot_ret < COV_GATE * tot_out
        all_pairs += [(i, j, partial, resid) for i, j, resid in pa]
    if not all_pairs:
        return {"not_evaluable": True, "reasons": list(set(ne_reasons)), "type": mode}
    # collect frames, extract DINOv2 (cls + patch)
    need = sorted(set(x for p in all_pairs for x in p[:2]))
    pos = {v: k for k, v in enumerate(need)}
    fr = vr.get_batch([min(i, NF-1) for i in need]).asnumpy()
    cls, pat = dino_feats_full(fr)
    d_cls=[]; d_pmean=[]; d_ppatch=[]; d_lp=[]
    for p in all_pairs:
        a, b = pos[p[0]], pos[p[1]]
        pm_a = F.normalize(pat[a].mean(0, keepdim=True), dim=-1); pm_b = F.normalize(pat[b].mean(0, keepdim=True), dim=-1)
        d_pmean.append(1 - float(F.cosine_similarity(pm_a, pm_b).item()))        # main metric
        d_cls.append(1 - float(F.cosine_similarity(cls[a:a+1], cls[b:b+1]).item()))
        d_ppatch.append(1 - float((pat[a]*pat[b]).sum(-1).mean().item()))
        d_lp.append(lpips_dist(fr[a], fr[b]))
    d_pmean = np.array(d_pmean)
    resid_arc = np.array([p[3] for p in all_pairs])
    ang_mis = float(np.mean(resid_arc)/deg2arc) if deg2arc else None       # rot: mean angle mismatch (degrees)
    # cls/ppatch/lpips (+ rot angle mismatch) silent archive (features already extracted, not tabled or documented)
    np.savez(f"{C2FEAT}/{model}__{stem}.npz", pmean=d_pmean, cls=np.array(d_cls),
             ppatch=np.array(d_ppatch), lpips=np.array(d_lp), resid_arc=resid_arc)
    dbar = float(np.mean(d_pmean))
    C2 = 100/(1 + dbar/TAU_C)
    per = 100/(1 + d_pmean/TAU_C)          # per-pair score, for +/- std
    out = {"not_evaluable": False, "type": mode, "n_pairs": len(all_pairs),
           "n_reversals_passed": len(revs) - len(ne_reasons),
           "partial": any(p[2] for p in all_pairs),
           "C2": round(C2, 3), "C2_std": round(float(np.std(per)), 3), "dbar": round(dbar, 4)}
    if mode == "rot": out["mean_angle_mismatch_deg"] = round(ang_mis, 2)
    return out


# ---------------- main ----------------
def canonical():
    # enumerate only by the flow-cache npz (no M1-M4 JSON dependency); full set includes all models (ALAYA/Genie3)
    ps = []
    for f in sorted(glob.glob(f"{CACHE}/*.npz")):
        m, stem = os.path.basename(f)[:-4].split("__")
        if os.path.exists(f"{BASE}/{m}/{stem}.mp4"): ps.append((m, stem))
    return ps

def global_delta(pairs):
    per = []
    for m, stem in pairs:
        z = np.load(f"{CACHE}/{m}__{stem}.npz", allow_pickle=True)
        mag = np.sqrt(z["Sx"]**2 + z["Srad"]**2); tm = z["tm"]
        dt = float(np.median(np.diff(tm))) if len(tm) > 1 else 1/FPS
        per.append(float(np.mean(mag)) * (DELTA_FRAC/dt))
    return float(np.median(per))

def main():
    SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
    allp = canonical(); pairs = [p for i, p in enumerate(allp) if i % NSHARD == SHARD]
    delta = global_delta(allp)
    print(f"videos: {len(pairs)}/{len(allp)} shard {SHARD}/{NSHARD} delta={delta:.2f}", flush=True)
    C2_ONLY = os.environ.get("C2_ONLY") == "1"
    C1_ONLY = os.environ.get("C1_ONLY") == "1"
    suffix = "_c2fix" if C2_ONLY else "_c1new" if C1_ONLY else ""
    outp = f"{OUT}/c1c2_v2{suffix}_shard{SHARD}of{NSHARD}.json"
    recs = json.load(open(outp)) if os.path.exists(outp) else {}
    for k, (m, stem) in enumerate(pairs):
        key = f"{m}__{stem}"
        if key in recs: continue
        try:
            vp = f"{BASE}/{m}/{stem}.mp4"; vr = VideoReader(vp, ctx=cpu(0))
            aid = load(m, stem)["aid"]
            if C2_ONLY:
                if aid not in (REVISIT | YAW_REV): continue   # only recompute C2 for translation + rotation round-trip videos
                c2 = c2_process(m, stem, vr)
                recs[key] = {"model": m, "stem": stem, "action": aid, "C2": c2}
                json.dump(recs, open(outp, "w"), default=float)
                print(f"[{k+1}/{len(pairs)}] {key} C2={c2.get('C2', c2.get('not_evaluable') and 'ne')} n={c2.get('n_pairs')}", flush=True)
            elif C1_ONLY:
                c1 = c1_process(m, stem, delta, vr); c1.update(c1_hardcut(vr))
                recs[key] = {"model": m, "stem": stem, "action": aid, "C1": c1}
                json.dump(recs, open(outp, "w"), default=float)
                print(f"[{k+1}/{len(pairs)}] {key} cut={c1.get('hard_cut')} lvl={c1.get('pmean_level')} mut={c1.get('mut_rate')} {c1.get('na','')}", flush=True)
            else:
                c1 = c1_process(m, stem, delta, vr); c1.update(c1_hardcut(vr))
                c2 = c2_process(m, stem, vr)
                recs[key] = {"model": m, "stem": stem, "action": aid, "C1": c1, "C2": c2}
                json.dump(recs, open(outp, "w"), default=float)
                print(f"[{k+1}/{len(pairs)}] {key} C1cut={c1.get('hard_cut')} C2={c2.get('C2', c2.get('na_scope') and 'scope' or c2.get('not_evaluable') and 'ne')}", flush=True)
        except Exception as e:
            print(f"[{k+1}/{len(pairs)}] FAIL {key}: {type(e).__name__} {e}", flush=True)
    print("DONE consistency_v2", flush=True)

if __name__ == "__main__":
    main()
