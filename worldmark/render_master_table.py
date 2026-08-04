"""Render final_master.csv into a colored table PNG (green = column max / red = column min)."""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wm_config as config

TAG = config.SUFFIX.lstrip("_") or "real_first"
rows=list(csv.DictReader(open(f"{config.quality_dir()}/final_master.csv")))
COLS=["DA_t","DA_r","MP_t","MP_r","MS_t","MS_r","RL_t","RL_r","Local","Global","Rev","QA_q","QA_a"]
HDR ={"DA_t":"Direction\nAccuracy\ntrans","DA_r":"Direction\nAccuracy\nrot",
      "MP_t":"Direction\nPurity\ntrans","MP_r":"Direction\nPurity\nrot",
      "MS_t":"Motion\nStability\ntrans","MS_r":"Motion\nStability\nrot",
      "RL_t":"Response\nLatency\ntrans","RL_r":"Response\nLatency\nrot",
      "Local":"Local\nMemory","Global":"Global\nMemory","Rev":"Revisit\nMemory",
      "QA_q":"Perceptual\nQuality","QA_a":"Aesthetic\nQuality"}
GROUP=["Action Dynamics"]*8 + ["World Memory"]*3 + ["Visual Quality"]*2
LOWER=set()
def fmt(c,v):                      # uniform [0,100], 2 decimals
    if v!=v: return "—"
    return f"{v:.2f}"
models=[r["model"] for r in rows]
val={c:{r["model"]:(float(r[c]) if r[c]!="" else np.nan) for r in rows} for c in COLS}
best={};second={};worst={}
for c in COLS:
    vv={m:val[c][m] for m in models if val[c][m]==val[c][m]}
    if not vv: continue
    order=sorted(vv,key=vv.get,reverse=(c not in LOWER))   # best to worst
    best[c]=order[0]; worst[c]=order[-1]
    if len(order)>=3: second[c]=order[1]                    # second place (only marked with >=3 models, to avoid overlapping with worst)

nrow=len(models)
MW=2.9                       # model-column width (multiple of a normal column)
CB=0.9                       # group-band height (row units)
HB=3.0                       # column-header height (row units, fits 3-line full names)
units=MW+len(COLS)           # horizontal units
vunits=nrow+CB+HB            # vertical units
fig,ax=plt.subplots(figsize=(1.05*units+0.6, 0.52*nrow+2.0)); ax.axis("off")
cw=1.0/units; w0=MW*cw; rh=1.0/vunits; cbh=CB*rh; hh=HB*rh
def xof(j): return 0 if j==0 else w0+(j-1)*cw   # left edge of column j (j=0 is model column)
def wof(j): return w0 if j==0 else cw
GREEN="#7fca7f"; YELLOW="#f5df7a"; RED="#f28c8c"; HEAD="#33373f"; ALT="#f4f5f7"
GCOL={"Action Dynamics":"#3a5a78","World Memory":"#5a4a78","Visual Quality":"#78603a"}
# top group band (merge consecutive same-group columns)
j=0
while j < len(COLS):
    k=j
    while k+1 < len(COLS) and GROUP[k+1]==GROUP[j]: k+=1
    x0=xof(j+1); x1=xof(k+1)+cw; g=GROUP[j]
    ax.add_patch(plt.Rectangle((x0,1-cbh),x1-x0,cbh,facecolor=GCOL[g],edgecolor="white",lw=1.5))
    ax.text((x0+x1)/2,1-cbh/2,g,ha="center",va="center",color="white",fontsize=10,fontweight="bold")
    j=k+1
ax.add_patch(plt.Rectangle((0,1-cbh),w0,cbh,facecolor="#222",edgecolor="white",lw=1.5))
# column headers (full names, taller)
ytop=1-cbh
for jj,c in enumerate(["model"]+COLS):
    ax.add_patch(plt.Rectangle((xof(jj),ytop-hh),wof(jj),hh,facecolor=HEAD,edgecolor="white",lw=1))
    ax.text(xof(jj)+wof(jj)/2,ytop-hh/2,("model" if c=="model" else HDR[c]),ha="center",va="center",
            color="white",fontsize=7.6,fontweight="bold",linespacing=1.2)
# data rows
for i,m in enumerate(models):
    y=ytop-hh-(i+1)*rh
    ax.add_patch(plt.Rectangle((0,y),w0,rh,facecolor=(ALT if i%2 else "white"),edgecolor="#dddddd",lw=.5))
    ax.text(w0*0.03,y+rh/2,m,ha="left",va="center",fontsize=9,fontweight="bold")
    for jj,c in enumerate(COLS):
        x=xof(jj+1); v=val[c][m]
        bg=(ALT if i%2 else "white")
        if best.get(c)==m: bg=GREEN
        elif second.get(c)==m: bg=YELLOW
        elif worst.get(c)==m: bg=RED
        ax.add_patch(plt.Rectangle((x,y),cw,rh,facecolor=bg,edgecolor="#dddddd",lw=.5))
        ax.text(x+cw/2,y+rh/2,fmt(c,v),ha="center",va="center",fontsize=8.4,
                color=("#999" if v!=v else "black"))
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.title(f"World-Model Benchmark — {TAG} (~125 videos/model)   ·   all metrics [0,100], higher is better   ·   green = 1st, yellow = 2nd, red = worst (per column)",
          fontsize=11, pad=10)
out=f"{config.RESULTS_ROOT}/master_table_{TAG}.png"
plt.savefig(out,dpi=200,bbox_inches="tight",facecolor="white"); print("saved",out)
