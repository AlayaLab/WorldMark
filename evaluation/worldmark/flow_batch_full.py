"""Full M1-M4 run: all videos under the video root (including Genie3/ALAYA), optical flow + depth parallax.
Multi-GPU sharding via env SHARD/NSHARD. Skips already-cached items (if a cache exists, only re-score).
Naming: {model}__{stem} (unique within the video root)."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm_config as config
import flow_eval as FE

BASE = os.environ.get('EVAL_BASE', config.VIDEO_ROOT)
OUT = config.flow_dir()
CACHE = config.flow_cache(); os.makedirs(CACHE, exist_ok=True)

tasks = []
for m in sorted(os.listdir(BASE)):
    if not os.path.isdir(f'{BASE}/{m}'): continue
    for p in sorted(glob.glob(f'{BASE}/{m}/*.mp4')):
        stem = os.path.basename(p)[:-4]
        try: a = int(stem.split('_')[1])
        except Exception: continue          # skip entries whose action id cannot be parsed
        tasks.append((m, stem, a, p))

SHARD = int(os.environ.get('SHARD', '0')); NSHARD = int(os.environ.get('NSHARD', '1'))
mine = [t for i, t in enumerate(tasks) if i % NSHARD == SHARD]
done = sum(1 for m, s, a, p in mine if os.path.exists(f'{OUT}/{m}__{s}.json'))
print(f"[shard {SHARD}/{NSHARD}] {len(mine)}/{len(tasks)} tasks, {done} already done", flush=True)

for k, (model, stem, a, p) in enumerate(mine):
    outp = f'{OUT}/{model}__{stem}.json'; cp = f'{CACHE}/{model}__{stem}.npz'
    if os.path.exists(outp): continue                 # already done -> skip (resumable)
    try:
        if os.path.exists(cp):
            cache = FE.load_cache(cp); tag = 'cached'
        else:
            cache = FE.perceive(p, model, a); FE.save_cache(cache, cp); tag = 'perceived'
        rec = {'model': model, 'stem': stem, 'action': a, **FE.score(cache)}
        json.dump(rec, open(outp, 'w'), default=float)
        if (k + 1) % 10 == 0: print(f"[{k+1}/{len(mine)}] OK ({tag}) {model} {stem}", flush=True)
    except Exception as e:
        print(f"[{k+1}/{len(mine)}] FAIL {model} {stem}: {type(e).__name__} {e}", flush=True)
print(f"[shard {SHARD}] DONE", flush=True)
