"""Q-Align (OneAlign) quality + aesthetics over the canonical set (1fps video mode).

Dual-GPU sharding via SHARD/NSHARD. Writes results_quality/qalign_shard{S}of{N}.json.
"""
import os, sys, glob, json
import wm_config as config
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from decord import VideoReader, cpu

RF = config.flow_dir()
BASE = config.VIDEO_ROOT
OUT = config.quality_dir()
os.makedirs(OUT, exist_ok=True)

_m = None
def model():
    global _m
    if _m is None:
        _m = AutoModelForCausalLM.from_pretrained(config.QALIGN_MODEL, trust_remote_code=True,
                                                  torch_dtype=torch.float16, device_map="cuda:0")
    return _m

def frames_1fps(path):
    vr = VideoReader(path, ctx=cpu(0)); fps = vr.get_avg_fps() or 16.0
    step = max(1, int(round(fps)))
    return [Image.fromarray(vr[i].asnumpy()) for i in range(0, len(vr), step)]

def canonical_pairs():
    ps = []
    for f in sorted(glob.glob(f"{RF}/*.json")):
        m, stem = os.path.basename(f)[:-5].split("__")
        if os.path.exists(f"{BASE}/{m}/{stem}.mp4"): ps.append((m, stem))
    return ps

def main():
    SHARD = int(os.environ.get("SHARD", "0")); NSHARD = int(os.environ.get("NSHARD", "1"))
    allp = canonical_pairs(); pairs = [p for i, p in enumerate(allp) if i % NSHARD == SHARD]
    print(f"videos: {len(pairs)}/{len(allp)} (shard {SHARD}/{NSHARD})", flush=True)
    outp = f"{OUT}/qalign_shard{SHARD}of{NSHARD}.json"
    out = json.load(open(outp)) if os.path.exists(outp) else {}
    md = model()
    for k, (m, stem) in enumerate(pairs):
        key = f"{m}__{stem}"
        if key in out: continue
        try:
            fr = frames_1fps(f"{BASE}/{m}/{stem}.mp4")
            q = float(md.score([fr], task_="quality", input_="video"))
            a = float(md.score([fr], task_="aesthetics", input_="video"))
            out[key] = {"model": m, "stem": stem, "qalign_quality": q, "qalign_aesthetic": a, "n_frames": len(fr)}
            json.dump(out, open(outp, "w"))
            print(f"[{k+1}/{len(pairs)}] {key} q={q:.3f} a={a:.3f}", flush=True)
        except Exception as e:
            print(f"[{k+1}/{len(pairs)}] FAIL {key}: {type(e).__name__} {e}", flush=True)
    print("DONE qalign_batch", flush=True)

if __name__ == "__main__":
    main()
