#!/usr/bin/env python3
"""Mechanically validate a delivered set of benchmark videos against the contract.

This is the acceptance gate. It does not trust any manifest — it decodes the mp4s.

    python check_delivery.py --arena arena_inputs/ \
        --delivery real/first/ALAYAWORLD --view first_view --domain real

Exit code 0 = accepted, 1 = rejected. Checks:
  1. filenames match {image:03d}_{action:03d}.mp4
  2. exactly the (image, action) pairs assigned in {view}/{domain}_action.txt, no extras
  3. every file decodes
  4. duration within tolerance of  segments x --sec-per-segment
  5. resolution and fps are internally consistent across the delivery
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

NAME_RE = re.compile(r"^(\d{3})_(\d{3})\.mp4$")
_FFPROBE_TIMEOUT = 20   # a broken ffmpeg can hang on OS crash reporting


def parse_action_protocol(path: Path) -> dict[int, str]:
    """action_id -> key string, e.g. 13 -> 'WRW'. Lines look like ' 13: WRW  (...)'."""
    out: dict[int, str] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(\d+):\s*([WSADLR]+)\b", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    if not out:
        sys.exit(f"could not parse any actions from {path}")
    return out


def parse_assignments(path: Path) -> dict[int, list[int]]:
    """row i (0-based) -> list of action ids assigned to image i."""
    out: dict[int, list[int]] = {}
    for i, line in enumerate(l for l in path.read_text().splitlines() if l.strip()):
        out[i] = [int(x) for x in re.split(r"[,\s]+", line.strip()) if x]
    return out


def _probe_ffprobe(path: Path) -> dict | None:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,duration",
           "-count_packets", "-of", "json", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT)
        st = json.loads(r.stdout)["streams"][0]
    except Exception:
        return None
    num, _, den = st.get("r_frame_rate", "0/1").partition("/")
    try:
        fps = float(num) / float(den) if float(den) else 0.0
    except ValueError:
        fps = 0.0
    frames = int(st.get("nb_read_packets") or 0)
    dur = st.get("duration")
    sec = float(dur) if dur not in (None, "N/A") else (frames / fps if fps else 0.0)
    if not frames or not fps:
        return None
    return {"width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
            "fps": round(fps, 3), "frames": frames, "sec": round(sec, 2)}


def _probe_cv2(path: Path) -> dict | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if not frames or not fps:
        return None
    return {"width": w, "height": h, "fps": round(fps, 3), "frames": frames,
            "sec": round(frames / fps, 2)}


_BACKENDS = {"ffprobe": _probe_ffprobe, "cv2": _probe_cv2}


def pick_backend(sample: Path, forced: str = "auto") -> tuple[str, callable]:
    """Choose a probe backend ONCE, against one known-good file.

    Deliberately not per-file: a broken ffmpeg install (e.g. a missing codec dylib)
    makes every ffprobe call cost minutes because the OS crash reporter kicks in, so
    a per-file fallback chain would hang the whole gate. Detect once, then commit.
    """
    if forced != "auto":
        return forced, _BACKENDS[forced]
    for name, fn in _BACKENDS.items():
        try:
            if fn(sample) is not None:
                return name, fn
        except Exception:
            pass
    sys.exit("no working video probe: install a working ffprobe, or `pip install "
             "opencv-python-headless`, or pass --probe-backend")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arena", required=True, type=Path, help="arena_inputs/ root")
    p.add_argument("--delivery", required=True, type=Path, help="dir holding the mp4s")
    p.add_argument("--view", required=True, choices=["first_view", "third_view"])
    p.add_argument("--domain", required=True, choices=["real", "style"])
    p.add_argument("--sec-per-segment", type=float, default=20.0)
    p.add_argument("--tolerance", type=float, default=10.0, help="percent, default 10")
    p.add_argument("--probe-backend", choices=["auto", "ffprobe", "cv2"],
                   default="auto")
    p.add_argument("--json-out", type=Path, default=None)
    a = p.parse_args()

    proto = parse_action_protocol(a.arena / "action_protocol.txt")
    assign = parse_assignments(a.arena / a.view / f"{a.domain}_action.txt")
    expected = {(img, act) for img, acts in assign.items() for act in acts}

    files = sorted(a.delivery.glob("*.mp4"))
    if not files:
        sys.exit(f"no .mp4 found in {a.delivery}")
    backend_name, probe_fn = pick_backend(files[0], a.probe_backend)
    errors: list[str] = []
    warnings: list[str] = []

    # 1. names
    found: dict[tuple[int, int], Path] = {}
    for f in files:
        m = NAME_RE.match(f.name)
        if not m:
            errors.append(f"bad filename: {f.name} (want {{image:03d}}_{{action:03d}}.mp4)")
            continue
        found[(int(m.group(1)), int(m.group(2)))] = f

    # 2. completeness
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    for img, act in missing:
        errors.append(f"missing: {img:03d}_{act:03d}.mp4")
    for img, act in extra:
        errors.append(f"not in assignment: {img:03d}_{act:03d}.mp4")

    # 3-4. decode + duration
    rows, res_set, fps_set = [], set(), set()
    for (img, act), f in sorted(found.items()):
        if (img, act) not in expected:
            continue
        info = probe_fn(f)
        if info is None:
            errors.append(f"undecodable: {f.name}")
            continue
        keys = proto.get(act, "")
        target = len(keys) * a.sec_per_segment
        dev = (info["sec"] - target) / target * 100 if target else 0.0
        ok = abs(dev) <= a.tolerance
        if not ok:
            errors.append(f"duration {f.name}: {info['sec']}s vs target {target}s "
                          f"({dev:+.1f}%, tolerance +/-{a.tolerance}%)")
        res_set.add((info["width"], info["height"]))
        fps_set.add(info["fps"])
        rows.append({"image": f"{img:03d}", "action": act, "keys": keys,
                     "target_sec": target, **info, "dev_pct": round(dev, 1),
                     "ok": ok})

    # 5. internal consistency
    if len(res_set) > 1:
        warnings.append(f"mixed resolutions in one delivery: {sorted(res_set)}")
    if len(fps_set) > 1:
        warnings.append(f"mixed fps in one delivery: {sorted(fps_set)}")

    # report
    print(f"delivery : {a.delivery}   (probe backend: {backend_name})")
    print(f"suite    : {a.view} / {a.domain}   expected {len(expected)} videos, "
          f"found {len(found)}")
    if res_set:
        print(f"measured : {sorted(res_set)} @ {sorted(fps_set)} fps")
    if rows:
        by_seg: dict[int, list[dict]] = {}
        for r in rows:
            by_seg.setdefault(len(r["keys"]), []).append(r)
        print(f"\n{'segs':>4} {'n':>4} {'target':>8} {'median':>8} {'min':>8} {'max':>8}")
        for nseg in sorted(by_seg):
            g = by_seg[nseg]
            secs = sorted(r["sec"] for r in g)
            print(f"{nseg:>4} {len(g):>4} {nseg*a.sec_per_segment:>8.1f} "
                  f"{secs[len(secs)//2]:>8.2f} {secs[0]:>8.2f} {secs[-1]:>8.2f}")

    for w in warnings:
        print(f"\nWARN  {w}")
    if errors:
        print(f"\nREJECTED — {len(errors)} error(s):")
        for e in errors[:40]:
            print(f"  - {e}")
        if len(errors) > 40:
            print(f"  ... and {len(errors)-40} more")
    else:
        print("\nACCEPTED")

    if a.json_out:
        a.json_out.write_text(json.dumps(
            {"delivery": str(a.delivery), "view": a.view, "domain": a.domain,
             "expected": len(expected), "found": len(found),
             "resolutions": sorted(res_set), "fps": sorted(fps_set),
             "errors": errors, "warnings": warnings, "videos": rows}, indent=2))

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
