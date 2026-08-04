"""Shared: enumerate all benchmark videos. id = domain__view__model__stem."""
import os, glob
import wm_config as config

ROOT = config.VIDEO_ROOT
RES = config.RESULTS_ROOT
HUD = {"SANA-WM-streaming": [0.18, 0.22]}   # only SANA has a baked-in WASD HUD overlay


def list_videos():
    out = []
    for dom in ("real", "style"):
        for view in ("first", "third"):
            for mp in sorted(glob.glob(f"{ROOT}/{dom}/{view}/*/*.mp4")):
                model = os.path.basename(os.path.dirname(mp))
                stem = os.path.basename(mp)[:-4]
                try:
                    action = int(stem.split("_")[1])
                except Exception:
                    action = None
                vid = f"{dom}__{view}__{model}__{stem}"
                out.append({"id": vid, "domain": dom, "view": view, "model": model,
                            "stem": stem, "action": action, "path": mp})
    return out


def shard(items, idx, n):
    return items[idx::n] if n > 1 else items
