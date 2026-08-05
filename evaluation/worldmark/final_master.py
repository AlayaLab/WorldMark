"""Master table: M1-M4 + Q-Align (quality/aesthetic) + C1 (local) + C2 (revisit) + C3 (global)."""
import json, glob, os, numpy as np
from collections import defaultdict
import wm_config as config
EXCLUDE = {"Open-Oasis", "Genie3"}                     # Open-Oasis too weak / Genie3 too few samples (5), not statistically meaningful
FAM = {"W":"fwd","S":"fwd","A":"lat","D":"lat","L":"yaw","R":"yaw"}

recs=[json.load(open(f)) for f in glob.glob(f"{config.flow_dir()}/*.json")]
bym=defaultdict(list)
for r in recs: bym[r["model"]].append(r)
def m1(rs,fam): v=[s.get("dir_gated",s.get("dir")) for r in rs for s in r.get("M1",[]) if not s.get("na") and FAM.get(s.get("key"))==fam and s.get("dir_gated",s.get("dir")) is not None]; return float(np.mean(v)) if v else np.nan
def m2(rs,fam): v=[s["purity"] for r in rs for s in r.get("M2",[]) if not s.get("na") and "purity" in s and s.get("family")==fam]; return float(np.mean(v)) if v else np.nan
def scr(rs,mk): v=[s["score"] for r in rs for s in r.get(mk,[]) if not s.get("na") and "score" in s]; v=[x for x in v if 0<=x<=1]; return float(np.mean(v)) if v else np.nan
# Pool by family (aggregate per-segment / per-response-point, divide by total segment count -- not mean of two group means)
def m1g(rs,fs): v=[s.get("dir_gated",s.get("dir")) for r in rs for s in r.get("M1",[]) if not s.get("na") and FAM.get(s.get("key")) in fs and s.get("dir_gated",s.get("dir")) is not None]; return float(np.mean(v)) if v else np.nan
def m2g(rs,fs): v=[s["purity"] for r in rs for s in r.get("M2",[]) if not s.get("na") and "purity" in s and s.get("family") in fs]; return float(np.mean(v)) if v else np.nan
def m4g(rs,fs): v=[s["score"] for r in rs for s in r.get("M4",[]) if not s.get("na") and "score" in s and FAM.get(s.get("key")) in fs]; v=[x for x in v if 0<=x<=1]; return float(np.mean(v)) if v else np.nan
def m3g(rs,fs): v=[s["score"] for r in rs for s in r.get("M3",[]) if not s.get("na") and "score" in s and FAM.get(s.get("new")) in fs]; v=[x for x in v if 0<=x<=1]; return float(np.mean(v)) if v else np.nan
TR={"fwd","lat"}; RO={"yaw"}

qa={}
for sh in glob.glob(f"{config.quality_dir()}/qalign_shard*.json"): qa.update(json.load(open(sh)))
qb=defaultdict(list)
for r in qa.values(): qb[r["model"]].append((r["qalign_quality"],r["qalign_aesthetic"]))
c3={}
for sh in glob.glob(f"{config.c3_dir()}/omega_c3_1s_shard*.json"): c3.update(json.load(open(sh)))
cb=defaultdict(list)
for r in c3.values():
    if "global_median_e" in r: cb[r["model"]].append(r["global_median_e"])
# C1: c1new shard (pmean adjacent-frame jumps); C2: c2fix shard (matched equal-motion pairs + patch-mean main score)
cc={}
for sh in glob.glob(f"{config.consistency_dir()}/c1c2_v2_c1new_shard*.json"): cc.update(json.load(open(sh)))
c1b=defaultdict(list)
for r in cc.values():
    if "C1" in r: c1b[r["model"]].append(r["C1"])
ccf={}
for sh in glob.glob(f"{config.consistency_dir()}/c1c2_v2_c2fix_shard*.json"): ccf.update(json.load(open(sh)))
TRANS_A={6,8,11,14}; ROT_A={7,12}
c2t=defaultdict(list); c2r=defaultdict(list)   # translation column / rotation column, reported separately (not combined)
for r in ccf.values():
    c2=r["C2"]
    if c2.get("na_scope") or c2.get("not_evaluable"): continue
    (c2t if r["action"] in TRANS_A else c2r)[r["model"]].append(c2["C2"])

def mean(xs): xs=[x for x in xs if x is not None]; return float(np.mean(xs)) if xs else np.nan
TAU_MUT=0.10   # consistent with consistency_v2 (frozen value, do not recompute P90 each run)
CUT_GUARD=8    # exclude frame pairs within +/-0.5s (@16fps) of hard cuts, to avoid double-penalizing with the hard-cut factor
def c1_score(cs):
    # Pure temporal: (1-hard_cut_rate)*(1-jump_rate); both factors are computed ONLY on non-frozen videos
    # (frozen videos do not enter C1, they are penalized by the M-series metrics).
    ev=[c for c in cs if not c.get("na") and c.get("pmean_dists") is not None]
    if not ev: return np.nan                      # all frozen -> C1 not_evaluable
    hc=mean([1 if c.get("hard_cut") else 0 for c in ev])
    muts=[]
    for c in ev:
        dists=np.array(c["pmean_dists"]); idx=c.get("idx",[]); cuts=c.get("cut_frames") or []
        keep=np.ones(len(dists),bool)
        if cuts and len(idx)==len(dists)+1:       # exclude frame pairs near hard cuts (4a)
            for k in range(len(dists)):
                if any(abs(idx[k]-cf)<=CUT_GUARD or abs(idx[k+1]-cf)<=CUT_GUARD for cf in cuts): keep[k]=False
        d=dists[keep]
        if len(d): muts.append(float(np.mean(d>TAU_MUT)))
    if not muts: return np.nan
    return 100*(1-(hc if not np.isnan(hc) else 0))*(1-float(np.mean(muts)))

# Order: Action Dynamics (6) -> World Memory (4, order Local/Global/Revisit) -> Visual Quality (2)
# M1/M2 fwd,lat merged into trans (mean), yaw->rot; Response Latency keeps only seconds; Local/Revisit normalized to [0,1]
# AD internal order: Direction -> Purity -> MotionStability -> ResponseLatency, each split trans/rot; WM: Local -> Global -> Revisit (merged)
COLS=["DA_t","DA_r","MP_t","MP_r","MS_t","MS_r","RL_t","RL_r","Local","Global","Rev","QA_q","QA_a"]
LOWER=set()                  # all normalized to [0,100], higher is better (Response Latency also reported as score 100*(1-lat/L))
TAU_G=0.05                   # Global Memory normalization scale: 100/(1+e/tau), tau ~ cross-domain median error (rounded); e = median reprojection error
NAME_MAP={"ALAYAWORLD":"AlayaWorld","DreamX-World-AR":"DreamX-World","HY-GameCraft":"HY-GameCraft 1.0",
          "HY-World":"HY-World 1.5","LingBotWorld-Fast":"LingBot-World","Lyra-2":"Lyra 2.0",
          "MatrixGame2.0":"Matrix-Game 2.0","MatrixGame3.0":"Matrix-Game 3.0",
          "SANA-WM-streaming":"SANA-WM","Yume":"Yume 1.5"}
def avg(*xs): xs=[x for x in xs if x==x]; return float(np.mean(xs)) if xs else np.nan
def s100(x):  return 100.0*x if x==x else np.nan                 # [0,1] -> [0,100]
def da100(x): return 100.0*max(0.0,x) if x==x else np.nan        # direction in [-1,1], negative (reversed) clipped to 0
def qa100(x): return 100.0*(x-1.0)/4.0 if x==x else np.nan       # Q-Align 1-5 -> [0,100]
rows={}
for m in bym:
    if m in EXCLUDE: continue
    disp=NAME_MAP.get(m,m)
    revall=(c2t.get(m,[])+c2r.get(m,[]))                         # Revisit merged: pool over all round-trip videos
    rows[disp]={"DA_t":da100(m1g(bym[m],TR)),"DA_r":da100(m1g(bym[m],RO)),
                "MP_t":s100(m2g(bym[m],TR)),"MP_r":s100(m2g(bym[m],RO)),
                "MS_t":s100(m4g(bym[m],TR)),"MS_r":s100(m4g(bym[m],RO)),
                "RL_t":s100(m3g(bym[m],TR)),"RL_r":s100(m3g(bym[m],RO)),   # score 100*(1-lat/L), higher is better
                "Local":c1_score(c1b.get(m,[])),                 # already 0-100
                "Global":s100(1.0/(1.0+mean(cb[m])/TAU_G)) if cb.get(m) else np.nan,
                "Rev":mean(revall) if revall else np.nan,        # C2 already 0-100
                "QA_q":qa100(mean([x[0] for x in qb[m]])) if qb.get(m) else np.nan,
                "QA_a":qa100(mean([x[1] for x in qb[m]])) if qb.get(m) else np.nan}
best={};worst={}
for c in COLS:
    vv={m:rows[m][c] for m in rows if not np.isnan(rows[m][c])}
    if not vv: continue
    if c in LOWER: best[c]=min(vv,key=vv.get); worst[c]=max(vv,key=vv.get)
    else: best[c]=max(vv,key=vv.get); worst[c]=min(vv,key=vv.get)
short={"DA_t":"Dir-trans","DA_r":"Dir-rot","MP_t":"Purity-trans","MP_r":"Purity-rot","MS_t":"Stab-trans","MS_r":"Stab-rot",
       "RL_t":"Lat-trans","RL_r":"Lat-rot","Local":"Local","Global":"Global","Rev":"Revisit","QA_q":"Percept","QA_a":"Aesth"}
hdr=f"{'model':<18}"+"".join(f"{short[c]:>10}" for c in COLS)
print(hdr); print("-"*len(hdr))
for m in sorted(rows):
    line=f"{m:<18}"
    for c in COLS:
        v=rows[m][c]
        if np.isnan(v): line+=f"{'—':>10}"; continue
        tag="+" if best.get(c)==m else("-" if worst.get(c)==m else " ")
        line+=f"{('%.2f'%v):>9}{tag}"
    print(line)
print("\nAll [0,100], higher is better (2 decimals); + best, - worst; trans=fwd+lat segments pooled, rot=yaw segments; Response Latency=100*(1-lat/L); Revisit=round-trip videos merged; QA=(x-1)/4*100; Global=100/(1+e/0.05)")
with open(f"{config.quality_dir()}/final_master.csv","w") as f:
    f.write("model,"+",".join(COLS)+"\n")
    for m in sorted(rows): f.write(m+","+",".join(f"{rows[m][c]:.4f}" if not np.isnan(rows[m][c]) else "" for c in COLS)+"\n")
print(f"CSV -> {config.quality_dir()}/final_master.csv")
