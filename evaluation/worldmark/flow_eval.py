"""M1-M4 optical-flow (SEA-RAFT) + depth-parallax (DA3) version, unified segmentation.
Metrics: M1 direction | M2 purity | M3 latency | M4 speed stability.
Signals: adjacent-frame optical flow -> Sx (horizontal, >0 = left yaw / left translation), Srad (radial, >0 = forward).
Direction is taken per key family: fwd (W/S) -> Srad; lat (A/D) / yaw (L/R) -> Sx.
Depth parallax: fit u = a/Z + b over lat/yaw segments -> soft likelihoods p_trans / p_yaw,
used for M1 type weighting + M2 purity axis decomposition.
Unified segmentation: boundaries at k/n; transition zone [t0-1s, t0+3s]; stable zone = segment minus transition.
  M1/M2/M4 use the stable zone; M3 anchors to the switch point t0.
"""
import sys, os, numpy as np, cv2, torch
import torch.nn.functional as F
import wm_config as config
# SEA-RAFT repo must be importable; derive its root from the config json path.
_SEARAFT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(config.SEARAFT_CFG)))
sys.path.insert(0, _SEARAFT_ROOT); sys.path.append(os.path.join(_SEARAFT_ROOT, 'core'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P

FPS = 16.0
TRANS_PRE_S, TRANS_POST_S = 1.0, 3.0
KEY_FLOW = {"W": ("rad", +1), "S": ("rad", -1), "A": ("hor", +1), "D": ("hor", -1),
            "L": ("hor", +1), "R": ("hor", -1)}
KEY_FAMILY = {"W": "fwd", "S": "fwd", "A": "lat", "D": "lat", "L": "yaw", "R": "yaw"}
REVERSAL = {6, 7, 8, 11, 12, 14}
HUD = {'SANA-WM-streaming': (0.18, 0.22)}

_RAFT = _DA3 = _RARGS = None


def _load():
    global _RAFT, _DA3, _RARGS
    if _RAFT is None:
        import argparse
        from config.parser import parse_args
        from raft import RAFT
        p = argparse.ArgumentParser(); p.add_argument('--cfg'); p.add_argument('--path', default=None)
        p.add_argument('--url', default='MemorySlices/Tartan-C-T-TSKH-spring540x960-M'); p.add_argument('--device', default='cuda')
        sys.argv = ['x', '--cfg', config.SEARAFT_CFG]
        _RARGS = parse_args(p)
        _RAFT = RAFT.from_pretrained(_RARGS.url, args=_RARGS).to('cuda').eval()
        from depth_anything_3.api import DepthAnything3
        _DA3 = DepthAnything3.from_pretrained(config.DA3_MODEL).to('cuda').eval()
    return _RAFT, _DA3, _RARGS


@torch.no_grad()
def _flow(im1, im2):
    raft, _, ra = _load()[0], None, _load()[2]
    def t(x): return torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)[None].cuda()
    i1 = F.interpolate(t(im1), scale_factor=2**ra.scale, mode='bilinear', align_corners=False)
    i2 = F.interpolate(t(im2), scale_factor=2**ra.scale, mode='bilinear', align_corners=False)
    o = _RAFT(i1, i2, iters=ra.iters, test_mode=True); fl = o['flow'][-1]
    fl = F.interpolate(fl, size=(im1.shape[0], im1.shape[1]), mode='bilinear', align_corners=False)
    return fl[0, 0].cpu().numpy(), fl[0, 1].cpu().numpy()


def _read(video, model, RH=432, RW=640):
    cap = cv2.VideoCapture(video); fr = []
    hud = HUD.get(model)
    while True:
        ok, f = cap.read()
        if not ok: break
        im = cv2.resize(f, (RW, RH))
        if hud:
            fh, fw = hud; im[int(RH*(1-fh)):, :int(RW*fw)] = 128
        fr.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cap.release(); return fr


def compute_signals(video, model, step=2):
    fr = _read(video, model); N = len(fr)
    RH, RW = fr[0].shape[:2]
    cx, cy = RW/2, RH/2; yy, xx = np.mgrid[0:RH, 0:RW]
    rx = xx-cx; ry = yy-cy; rn = np.hypot(rx, ry)+1e-6; rx /= rn; ry /= rn
    Sx = np.zeros(N-step); Srad = np.zeros(N-step)
    for i in range(N-step):
        u, v = _flow(fr[i], fr[i+step])
        Sx[i] = u.mean(); Srad[i] = (u*rx+v*ry).mean()
    tm = np.arange(N-step)/FPS
    return dict(tm=tm, Sx=Sx, Srad=Srad, fr=fr, N=N, rx=rx, ry=ry, step=step)


def _sig_for(key, S):
    kind, sgn = KEY_FLOW[key]
    return (S['Srad'] if kind == 'rad' else S['Sx']), sgn


def _seg_win(j, n, dur):
    lo, hi = j*dur/n, (j+1)*dur/n
    a = lo + (TRANS_POST_S if j > 0 else 0.0)
    b = hi - (TRANS_PRE_S if j < n-1 else 0.0)
    return a, b


def sparc(mv, fs, padlevel=4, fc=10.0, amp_th=0.05):
    mv = np.asarray(mv, float)
    if len(mv) < 4 or np.all(mv == 0): return 0.0
    nfft = int(2**(np.ceil(np.log2(len(mv)))+padlevel))
    f = np.arange(0, fs, fs/nfft); Mf = np.abs(np.fft.fft(mv, nfft))
    if Mf.max() <= 0: return 0.0
    Mf /= Mf.max(); sel = f <= fc; f, Mf = f[sel], Mf[sel]
    inx = np.where(Mf >= amp_th)[0]
    if len(inx) < 2: return 0.0
    f, Mf = f[inx[0]:inx[-1]+1], Mf[inx[0]:inx[-1]+1]
    return float(-np.sum(np.sqrt((np.diff(f)/(f[-1]-f[0]))**2+np.diff(Mf)**2)))


def metrics(S, action_id):
    keys = [s[1] for s in P.segments(action_id)]; n = len(keys)
    tm, dur = S['tm'], S['tm'][-1]; aid = int(action_id)
    m1 = []; m4 = []
    for j, k in enumerate(keys):
        a, b = _seg_win(j, n, dur); m = (tm >= a) & (tm <= b)
        sig, sgn = _sig_for(k, S); v = sgn*sig[m]
        if v.size < 3:
            m1.append({'seg': j, 'key': k, 'na': True}); m4.append({'seg': j, 'key': k, 'na': True}); continue
        dirc = float(np.mean(v)/(np.mean(np.abs(sig[m]))+1e-9))
        m1.append({'seg': j, 'key': k, 'family': KEY_FAMILY[k], 'dir': round(dirc, 3)})
        # M4 speed stability = sustain (duration) x exp(-hr/tau) (hiccup frequency) x decay (last/first ratio), three orthogonal dimensions
        g = float(np.percentile(v, 70))      # baseline = normal motion speed (P70, not dragged down by failures)
        if g <= 1e-6:
            m4.append({'seg': j, 'key': k, 'sustain': 0.0, 'score': 0.0}); continue
        w = max(1, int(0.3*FPS)); vs = np.convolve(v, np.ones(w)/w, mode='same')
        thr_lo, thr_hi = 0.4*g, 0.5*g
        sustain = float(np.mean(vs >= thr_lo))               # fraction of time held at normal speed
        minlen = max(1, int(0.3*FPS)); ev = 0; in_low = False; run = 0   # hiccups: hysteresis counting
        for x in vs:
            if in_low:
                run += 1
                if x > thr_hi:
                    if run >= minlen: ev += 1
                    in_low = False; run = 0
            elif x < thr_lo:
                in_low = True; run = 1
        if in_low and run >= minlen: ev += 1
        hr = ev/max(v.size/FPS, 1e-6)
        h = max(1, int(3*FPS))                               # decay: last 3s / first 3s of the segment
        first = float(np.mean(vs[:h])); last = float(np.mean(vs[-h:]))
        decay = min(max(last/(first+1e-9), 0.0), 1.2)
        score = sustain * np.exp(-hr/0.5) * min(1.0, decay/0.7)
        m4.append({'seg': j, 'key': k, 'sustain': round(sustain, 3), 'hiccups': ev,
                   'decay': round(decay, 3), 'score': round(float(score), 3)})
    # M3 latency: absolute reference threshold (avoids self-referencing that would let weak responses off easy).
    # v_ref = P75 of the same channel's (Sx/Srad) whole-video steady speed magnitude = this model's "normal motion speed"; threshold = 0.5*v_ref.
    vref_x = float(np.percentile(np.abs(S['Sx']), 75)) + 1e-9
    vref_rad = float(np.percentile(np.abs(S['Srad']), 75)) + 1e-9
    m2 = []
    for j in range(1, n):
        k = keys[j]; t0 = j*dur/n; sig, sgn = _sig_for(k, S); v = sgn*sig
        kind = KEY_FLOW[k][0]; vref = vref_rad if kind == 'rad' else vref_x
        a, b = _seg_win(j, n, dur); sm = (tm >= a) & (tm <= b)
        ss = float(np.median(v[sm])) if sm.sum() > 2 else 0.0   # steady state (decides whether a real response exists)
        L = dur/n; thr = 0.5*vref
        # gate: if steady state has no genuine response toward the command direction (incl. reverse ss<0) -> score 0, prevents transient spikes being read as fast response
        if ss <= 0.1*vref:
            note = 'no_response' if ss <= 0 else 'weak_response'
            m2.append({'t0': round(t0, 1), 'new': k, 'latency': round(L, 2),
                       'ss_ratio': round(ss/vref, 3), 'score': 0.0, 'note': note}); continue
        w = max(1, int(0.5*FPS)); vs = np.convolve(v, np.ones(w)/w, mode='same')
        win = np.where((tm >= t0) & (tm <= t0+L))[0]
        lat = L
        for i in win:
            if vs[i] >= thr:               # reached "half of normal speed"
                lat = max(0.0, tm[i]-t0); break
        note = '' if lat < L else 'below_half_normal'
        m2.append({'t0': round(t0, 1), 'new': k, 'latency': round(lat, 2),
                   'ss_ratio': round(ss/vref, 3), 'score': round(1-min(lat, L)/L, 3),  # [0,1]
                   **({'note': note} if note else {})})
    # numbering: M1=direction, M3=latency, M4=speed stability (M2=purity is added in score())
    # (the former "reversal smoothness" metric was removed: under the mean-flow signal it cannot robustly
    #  distinguish smooth reversal from sluggish/jittery motion, and it overlapped with M1/M3/M4)
    return {'action_id': aid, 'keys': keys, 'M1': m1, 'M3': m2, 'M4': m4}


def _parallax_raw(S, action_id, nsamp=6):
    """Collect raw (invZ, u) points over lat/yaw segments (for caching). DA3 depth + optical-flow u."""
    from PIL import Image
    keys = [s[1] for s in P.segments(action_id)]; n = len(keys)
    fr = S['fr']; N = S['N']; step = S['step']
    _, da3, _ = _load()
    picks = []
    step1s = max(1, int(round(FPS)))                      # sample one frame per second (FPS=16)
    for j, k in enumerate(keys):
        if KEY_FAMILY[k] not in ('lat', 'yaw'): continue
        lo = int((j+0.25)*N/n); hi = int((j+0.9)*N/n)
        idxs = list(range(lo, min(hi, N-step-1), step1s))
        if len(idxs) < 3:                                 # fallback when the segment is too short
            idxs = [int(x) for x in np.linspace(lo, min(hi, N-step-1), 3)]
        picks += [(j, i) for i in idxs]
    raw = {}
    if not picks: return raw
    imgs = [Image.fromarray(fr[i]) for _, i in picks]
    dep = np.asarray(da3.inference(imgs).depth); DH, DW = dep.shape[1:]
    from collections import defaultdict
    bucket = defaultdict(list)
    for idx, (j, i) in enumerate(picks):
        u, _ = _flow(fr[i], fr[i+step]); u = cv2.resize(u, (DW, DH))
        invZ = 1.0/np.clip(dep[idx], 1e-3, None)
        iv = invZ.reshape(-1)[::6]; uu = u.reshape(-1)[::6]
        q1, q99 = np.percentile(iv, 1), np.percentile(iv, 99); kp = (iv >= q1) & (iv <= q99)
        bucket[j].append((iv[kp], uu[kp]))
    for j, lst in bucket.items():
        raw[j] = (np.concatenate([a for a, _ in lst]), np.concatenate([b for _, b in lst]))
    return raw


def parallax_score(raw, keys):
    """Decompose horizontal flow u = a/Z + b from cached raw points:
    trans_amp = |a| * (5-95% span of 1/Z) = parallax (translation) flow magnitude; rot_amp = |b| = far-field (rotation) flow magnitude.
    Soft likelihoods p_trans = trans_amp/(trans_amp+rot_amp), p_yaw = 1 - p_trans."""
    out = {}
    for j, (iv, uu) in raw.items():
        A = np.vstack([iv, np.ones_like(iv)]).T
        a, b = np.linalg.lstsq(A, uu, rcond=None)[0]
        spread = float(np.percentile(iv, 95) - np.percentile(iv, 5))
        trans_amp = abs(float(a)) * spread
        rot_amp = abs(float(b))
        p_trans = trans_amp / (trans_amp + rot_amp + 1e-6)
        out[int(j)] = {'a': round(float(a), 3), 'b_far': round(float(b), 3),
                       'trans_amp': round(trans_amp, 3), 'rot_amp': round(rot_amp, 3),
                       'p_trans': round(p_trans, 3), 'p_yaw': round(1.0 - p_trans, 3),
                       'family': KEY_FAMILY[keys[int(j)]]}
    return out


# ------------- perceive (compute once, cache flow+depth) / score (derive from cache) -------------
def perceive(video, model, action_id):
    """Compute once: SEA-RAFT optical flow (Sx, Srad) + DA3 depth-parallax raw points. Returns a cacheable dict (no frames)."""
    S = compute_signals(video, model)
    raw = _parallax_raw(S, action_id)
    return {'tm': S['tm'], 'Sx': S['Sx'], 'Srad': S['Srad'],
            'keys': [s[1] for s in P.segments(action_id)], 'action': int(action_id), 'px_raw': raw}


def save_cache(cache, path):
    flat = {'tm': cache['tm'], 'Sx': cache['Sx'], 'Srad': cache['Srad'],
            'keys': np.array(cache['keys']), 'action': cache['action'],
            'px_segs': np.array(sorted(cache['px_raw'].keys()), dtype=int)}
    for j, (iv, uu) in cache['px_raw'].items():
        flat[f'iv_{j}'] = iv; flat[f'uu_{j}'] = uu
    np.savez_compressed(path, **flat)


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    raw = {int(j): (z[f'iv_{j}'], z[f'uu_{j}']) for j in z['px_segs']}
    return {'tm': z['tm'], 'Sx': z['Sx'], 'Srad': z['Srad'],
            'keys': [str(k) for k in z['keys']], 'action': int(z['action']), 'px_raw': raw}


def motion_purity(S, action_id, PX):
    """Motion purity: per segment split motion into three channels -- lateral lat = p_trans*|Sx|, yaw = p_yaw*|Sx|, forward fwd = |Srad|;
    purity = commanded-axis component / (sum of the three) in [0,1]. 1 = pure commanded axis; low = off-axis leakage (e.g. moving forward while going left)."""
    keys = [s[1] for s in P.segments(action_id)]; n = len(keys)
    tm = S['tm']; dur = tm[-1]; out = []
    for j, k in enumerate(keys):
        a, b = _seg_win(j, n, dur); m = (tm >= a) & (tm <= b)
        if m.sum() < 3:
            out.append({'seg': j, 'key': k, 'na': True}); continue
        sx = float(np.mean(np.abs(S['Sx'][m]))); sr = float(np.mean(np.abs(S['Srad'][m])))
        fam = KEY_FAMILY[k]; px = PX.get(j)
        if fam in ('lat', 'yaw') and px:
            lateral = px['p_trans'] * sx; yaw = px['p_yaw'] * sx
        else:
            lateral = yaw = 0.5 * sx           # W/S has no depth split, the whole lateral component counts as off-axis
        fwd = sr; tot = lateral + yaw + fwd + 1e-9
        onaxis = {'lat': lateral, 'yaw': yaw, 'fwd': fwd}[fam]
        out.append({'seg': j, 'key': k, 'family': fam, 'purity': round(onaxis / tot, 3),
                    'lat': round(lateral, 2), 'fwd': round(fwd, 2), 'yaw': round(yaw, 2)})
    return out


def score(cache):
    """Compute the metrics + parallax from cache (cheap; changing metrics needs no re-perceive):
      M1 direction  (direction x motion-type soft likelihood: lat uses p_trans, yaw uses p_yaw, fwd unweighted)
      M2 purity  (commanded-axis component / sum of three axes)
      M3 latency  M4 reversal smoothness (SPARC)  M5 speed stability
    """
    S = {'tm': cache['tm'], 'Sx': cache['Sx'], 'Srad': cache['Srad']}
    M = metrics(S, cache['action'])            # contains M1, M3, M4, M5
    PX = parallax_score(cache['px_raw'], cache['keys'])
    for s in M['M1']:
        if s.get('na'):
            continue
        fam = KEY_FAMILY.get(s['key']); px = PX.get(s['seg'])
        w = px['p_trans'] if (fam == 'lat' and px) else (px['p_yaw'] if (fam == 'yaw' and px) else 1.0)
        s['type_w'] = round(w, 3)
        s['dir_gated'] = round(s['dir'] * w, 3)   # direction score after likelihood weighting
    M['M2'] = motion_purity(S, cache['action'], PX)   # M2 = purity
    M['parallax'] = {str(j): d for j, d in PX.items()}
    return M
