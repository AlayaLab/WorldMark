"""Direction accuracy v2 -- translation cosine + rotation axis-angle cosine, each normalized in its own space, scale-invariant.
Segmentation: split evenly by video length into n_seg parts (n_seg = number of action segments), not by a fixed 20s.
Input: uniformly sampled DA3 w2c extrinsics + per-frame position frac (0..1) + keys + action_id.
Frame-by-frame pairs within each part:
  translation: rotate Δt into the camera frame -> magnitude-weighted cosine against the expected axis
  rotation: per-frame axis-angle |ω|, net yaw = Σ Δψ (forward-vector method, right turn +); rot_dir = key_sign * netyaw / Σ|ω|
"""
import numpy as np

# camera frame [x=right, y=down, z=forward]
TRANS_AXIS = {"W": (0, 0, 1), "S": (0, 0, -1), "A": (-1, 0, 0), "D": (1, 0, 0)}
ROT_SIGN = {"R": +1, "L": -1}   # forward-vector yaw, right turn is positive


def _axis_angle_norm(R):
    """Return the rotation angle (radians)."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _centers_fwd_yaw(ext):
    """From w2c ext (M,3,4) compute: camera centers C (M,3), forward fwd (M,3), per-frame unwrapped yaw (M,)."""
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    C = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)
    fwd = R[:, 2, :]                       # camera z axis (forward) in world = 3rd row of the w2c R
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    return C, R, yaw


def _seg_scores(ext, key):
    """Frame-by-frame pairs within one part (segment) -> direction score for this key + diagnostics. ext: the part's w2c frames (m,3,4)."""
    if len(ext) < 2:
        return None
    C, R, yaw = _centers_fwd_yaw(ext)
    dtw = np.diff(C, axis=0)                          # per-frame translation in world frame
    dt_cam = np.einsum("nij,nj->ni", R[:-1], dtw)     # rotate into each frame's camera frame [right, down, forward]
    dpsi = np.diff(yaw)                               # per-frame yaw increment (rad, already unwrapped)
    omega = np.array([_axis_angle_norm(R[i + 1] @ R[i].T) for i in range(len(R) - 1)])  # per-frame total rotation angle
    out = {"key": key}
    if key in TRANS_AXIS:
        g = np.array(TRANS_AXIS[key], float)
        num = float(np.sum(dt_cam @ g))              # Σ Δt·gt
        den = float(np.sum(np.linalg.norm(dt_cam, axis=1)))  # Σ|Δt|
        out["dir"] = num / (den + 1e-12)             # magnitude-weighted cosine, scale-invariant
        net = float(np.linalg.norm(dt_cam.sum(0)))
        out["coherence"] = net / (den + 1e-12)       # net displacement / path length: low = jitter / frozen
        out["type"] = "trans"
    else:  # R / L
        s = ROT_SIGN[key]
        net_yaw = float(np.sum(dpsi))                # net yaw
        den = float(np.sum(np.abs(omega)))           # Σ total rotation angle (incl. off-axis)
        out["dir"] = s * net_yaw / (den + 1e-12)     # scale-invariant; off-axis / back-and-forth both reduce the score
        out["coherence"] = abs(net_yaw) / (float(np.sum(np.abs(dpsi))) + 1e-12)
        out["type"] = "rot"
    return out


def heading_composition(pose, edge_trim=0.2, turn_gate_deg=10.0):
    """§2.5 heading composition (action 13 WRW / 15 WRS), world frame.
    HC = cos( translation-heading change − camera self-rotation Δψ )  [13]; minus (Δψ+180°) [15].
    Translation heading and Δψ both use atan2(x,z) with the same handedness (right turn positive).
    seg0=W (t1), seg1=R (turn), seg2=W/S (t3). Gate: |Δψ|<10° -> not_evaluable."""
    aid = int(pose["action_id"])
    if aid not in (13, 15):
        return None
    ext = pose["ext"]; frac = np.asarray(pose["frac"]); keys = list(pose["keys"])
    if len(keys) != 3:
        return {"score": None, "note": "not 3-seg"}
    R = ext[:, :3, :3]; t = ext[:, :3, 3]
    C = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)  # world camera centers
    fwd = R[:, 2, :]                                             # w2c: camera forward (world) = 3rd row of R
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))            # right turn positive, radians

    def seg_mask(j, n=3):
        lo, hi = j / n, (j + 1) / n
        a = lo + (edge_trim / n if j > 0 else 0.0)
        b = hi - (edge_trim / n if j < n - 1 else 0.0)
        return (frac >= a - 1e-9) & (frac <= b + 1e-9)

    m0, m1, m2 = seg_mask(0), seg_mask(1), seg_mask(2)
    if m0.sum() < 2 or m1.sum() < 2 or m2.sum() < 2:
        return {"score": None, "note": "short seg"}
    C0, C2 = C[m0], C[m2]
    t1 = C0[-1] - C0[0]; t3 = C2[-1] - C2[0]          # net translation in world frame
    dpsi = float(yaw[m1][-1] - yaw[m1][0])            # seg1 self-rotation (rad, already unwrapped)
    if abs(np.degrees(dpsi)) < turn_gate_deg:
        return {"score": None, "note": "turn not triggered", "dpsi_deg": round(np.degrees(dpsi), 1)}
    a1, a3 = np.hypot(t1[0], t1[2]), np.hypot(t3[0], t3[2])   # horizontal displacement magnitude
    if a1 < 1e-9 or a3 < 1e-9:
        return {"score": None, "note": "no translation"}
    head1 = np.arctan2(t1[0], t1[2]); head3 = np.arctan2(t3[0], t3[2])  # same handedness as yaw
    head_change = (head3 - head1 + np.pi) % (2 * np.pi) - np.pi          # translation-heading change, wrapped
    ref = dpsi if aid == 13 else dpsi + np.pi                            # WRS: new heading reversed
    hc = float(np.cos(head_change - ref))
    return {"score": round(hc, 3), "dpsi_deg": round(np.degrees(dpsi), 1),
            "head_change_deg": round(np.degrees(head_change), 1)}


# ---- unified segmentation (shared by M1/M3/M4) ----
# boundaries use fractions k/n (tolerant to small length differences); transition zone = TRANS_PRE_S seconds before and TRANS_POST_S seconds after each switch.
# stable zone = segment minus transition zone. Each key lasts a fixed SEC_PER_KEY seconds -> seconds to in-segment fraction: s/SEC_PER_KEY * w.
SEC_PER_KEY = 20.0
TRANS_PRE_S = 1.0     # 1s before switch
TRANS_POST_S = 3.0    # 3s after switch


def evaluate(pose, edge_trim=None):
    """pose: dict{ext(M,3,4) w2c, frac(M,), keys[list], action_id}. Split evenly into n_seg parts by frac.
    Stable zone: at a segment's start (if preceded by a switch) drop 3s after the switch; at its end (if followed by a switch) drop 1s before the switch."""
    ext = pose["ext"]
    frac = np.asarray(pose["frac"])
    keys = list(pose["keys"])
    n = len(keys)
    w = 1.0 / n
    pre_f = (TRANS_PRE_S / SEC_PER_KEY) * w    # in-segment fraction corresponding to 1s
    post_f = (TRANS_POST_S / SEC_PER_KEY) * w  # 3s
    segs = []
    for j in range(n):
        lo, hi = j * w, (j + 1) * w
        # if a segment's start has a preceding switch -> drop 3s after; if its end has a following switch -> drop 1s before
        a = lo + (post_f if j > 0 else 0.0)
        b = hi - (pre_f if j < n - 1 else 0.0)
        m = (frac >= a - 1e-9) & (frac <= b + 1e-9)
        sub = ext[m]
        r = _seg_scores(sub, keys[j])
        if r is not None:
            r["seg"] = j
            r["dir"] = round(r["dir"], 4)
            r["coherence"] = round(r["coherence"], 3)
        segs.append(r if r is not None else {"seg": j, "key": keys[j], "na": True})
    return {"action_id": int(pose["action_id"]), "segments": segs}
