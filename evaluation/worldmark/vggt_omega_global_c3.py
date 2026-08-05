"""World Memory C3 Global via VGGT-Omega feed-forward reconstruction (deterministic).

Omega has no direct world_points output, so world_points are obtained by
back-projecting depth + pose, then cross-frame global consistency is computed
(identical protocol to the VGGT-1B version).
"""
import os, sys, glob, json, numpy as np
import wm_config as config
sys.path.insert(0, config.VGGT_OMEGA)
import torch, cv2
from decord import VideoReader, cpu
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

BASE = config.VIDEO_ROOT
CKPT = os.environ.get("VGGT_OMEGA_CKPT", "")
STRIDE_SEC = 1.0; MAX_FRAMES = 64; GRID_STEP = 16; OCC_LO, OCC_HI = 0.7, 1.4; CONF_Q = 0.3   # one frame per second (final version); drop lowest 30% of depth_conf
_m = None
def model():
    global _m
    if _m is None:
        _m = VGGTOmega().to("cuda").eval(); _m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    return _m


def samp(img, px, py):
    H, W = img.shape[:2]
    x0 = np.clip(np.floor(px).astype(int), 0, W - 1); y0 = np.clip(np.floor(py).astype(int), 0, H - 1)
    return img[y0, x0]


def process(model_name, stem, tag=""):
    vr = VideoReader(f"{BASE}/{model_name}/{stem}.mp4", ctx=cpu(0)); NF = len(vr)
    fps = vr.get_avg_fps() or 16.0
    step = max(1, int(round(STRIDE_SEC * fps)))
    fr = list(range(0, NF, step))
    if len(fr) < 6: fr = sorted(set(int(x) for x in np.linspace(0, NF - 1, 6)))         # fallback for very short clips
    if len(fr) > MAX_FRAMES: fr = [fr[i] for i in np.linspace(0, len(fr) - 1, MAX_FRAMES).astype(int)]
    td = f"/tmp/vo_g_{os.getpid()}_{tag}"; os.makedirs(td, exist_ok=True)   # per-process isolation so multiple shards don't collide
    for f in glob.glob(f"{td}/*.png"): os.remove(f)
    paths = []
    for i, f in enumerate(fr):
        p = f"{td}/{i:03d}.png"
        ok = cv2.imwrite(p, cv2.cvtColor(vr[f].asnumpy(), cv2.COLOR_RGB2BGR))
        if not ok or not os.path.exists(p):
            raise RuntimeError(f"imwrite failed frame {f}")
        paths.append(p)
    imgs = load_and_preprocess_images(paths, image_resolution=512).to("cuda")
    with torch.inference_mode():
        pred = model()(imgs)
    HW = pred["images"].shape[-2:]; H, W = HW
    extr, intr = encoding_to_camera(pred["pose_enc"], HW)
    extr = extr[0].float().cpu().numpy(); intr = intr[0].float().cpu().numpy()     # [K,3,4],[K,3,3]
    depth = pred["depth"][0, :, :, :, 0].float().cpu().numpy()                      # [K,H,W]
    conf = pred["depth_conf"][0].float().cpu().numpy()                             # [K,H,W]
    Kf = len(fr)
    # world_points by back-projecting depth + pose
    yy, xx = np.mgrid[0:H, 0:W]
    wp = np.zeros((Kf, H, W, 3), np.float32)
    for k in range(Kf):
        fx, fy, cx, cy = intr[k][0, 0], intr[k][1, 1], intr[k][0, 2], intr[k][1, 2]
        d = depth[k]
        Xc = np.stack([(xx - cx) / fx * d, (yy - cy) / fy * d, d], -1)            # cam
        R = extr[k][:, :3]; t = extr[k][:, 3]
        wp[k] = (Xc - t) @ R                                                       # world = R^T(Xc - t) => (Xc-t)@R
    scale = float(np.median(depth[depth > 1e-6]))
    conf_thr = np.quantile(conf, CONF_Q)
    gy, gx = yy[::GRID_STEP, ::GRID_STEP].ravel(), xx[::GRID_STEP, ::GRID_STEP].ravel()
    errs = []; covis = 0; total = 0
    for i in range(Kf):
        gm = conf[i][gy, gx] >= conf_thr                                          # use only high-confidence source points
        gyi, gxi = gy[gm], gx[gm]
        X = wp[i][gyi, gxi]
        for j in range(Kf):
            if i == j: continue
            R = extr[j][:, :3]; t = extr[j][:, 3]
            xc = X @ R.T + t; Z = xc[:, 2]; Kj = intr[j]
            px = Kj[0, 0] * xc[:, 0] / Z + Kj[0, 2]; py = Kj[1, 1] * xc[:, 1] / Z + Kj[1, 2]
            inside = (Z > 1e-6) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            if inside.sum() < 5: continue
            dj = samp(depth[j], px, py); ratio = Z / np.maximum(dj, 1e-6)
            cov = inside & (ratio > OCC_LO) & (ratio < OCC_HI) & (samp(conf[j], px, py) >= conf_thr)
            total += int(inside.sum()); covis += int(cov.sum())
            if cov.sum() < 5: continue
            Y = samp(wp[j], px[cov], py[cov])
            errs.append(np.linalg.norm(X[cov] - Y, axis=1) / scale)
    if not errs:
        return {"model": model_name, "stem": stem, "na": "no_covis"}
    alle = np.concatenate(errs)
    return {"model": model_name, "stem": stem,
            "global_median_e": round(float(np.median(alle)), 5), "global_mean_e": round(float(np.mean(alle)), 5),
            "global_score": round(100.0 / (1.0 + float(np.median(alle))), 3),
            "covis_ratio": round(covis / max(1, total), 3)}


def canonical_pairs():
    ps = []
    for f in sorted(glob.glob(f"{config.flow_dir()}/*.json")):
        m, stem = os.path.basename(f)[:-5].split("__")
        if os.path.exists(f"{BASE}/{m}/{stem}.mp4"): ps.append((m, stem))
    return ps


def main():
    OUTD = config.c3_dir(); os.makedirs(OUTD, exist_ok=True)
    SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
    allp = canonical_pairs(); pairs = [p for i, p in enumerate(allp) if i % NSHARD == SHARD]
    print(f"videos: {len(pairs)}/{len(allp)} (shard {SHARD}/{NSHARD})", flush=True)
    outp = f"{OUTD}/omega_c3_1s_shard{SHARD}of{NSHARD}.json"
    out = json.load(open(outp)) if os.path.exists(outp) else {}
    for m, stem in pairs:
        key = f"{m}__{stem}"
        if key in out: continue
        try:
            r = process(m, stem); out[key] = r; json.dump(out, open(outp, "w"), default=float)
            print(f"{key}: gmedE={r.get('global_median_e')} score={r.get('global_score')} covis={r.get('covis_ratio')}", flush=True)
        except Exception as e:
            print(f"FAIL {key}: {type(e).__name__} {e}", flush=True)
    print("DONE vggt_omega_global_c3", flush=True)


if __name__ == "__main__":
    main()
