"""Central configuration for WorldMark.

All paths are resolved from environment variables with sensible defaults, so the
code carries no machine-specific absolute paths. Override any of them by exporting
the corresponding environment variable before running.

Video layout expected:  {VIDEO_ROOT}/{model}/{stem}.mp4
  - `stem` encodes the action id as the second underscore field, e.g. "000_006" -> action 6.
Results layout (auto-created):  {RESULTS_ROOT}/results_<stage>{EVAL_SUFFIX}/
"""
import os

# ---- roots -------------------------------------------------------------------
ROOT         = os.environ.get("WORLDMARK_ROOT", os.path.dirname(os.path.abspath(__file__)))
VIDEO_ROOT   = os.environ.get("WORLDMARK_VIDEOS",  os.path.join(ROOT, "videos"))
RESULTS_ROOT = os.environ.get("WORLDMARK_RESULTS", os.path.join(ROOT, "results"))
# domain/view tag appended to every result dir so real/style, first/third stay separate
SUFFIX       = os.environ.get("EVAL_SUFFIX", "")

# ---- per-stage result directories -------------------------------------------
def flow_dir():         return f"{RESULTS_ROOT}/results_flow{SUFFIX}"           # M1-M4 (Action Dynamics)
def flow_cache():       return f"{flow_dir()}/cache"                            # SEA-RAFT + DA3 perceive cache (.npz)
def quality_dir():      return f"{RESULTS_ROOT}/results_quality{SUFFIX}"        # Q-Align (Visual Quality)
def consistency_dir():  return f"{RESULTS_ROOT}/results_consistency{SUFFIX}"    # C1 Local + C2 Revisit (World Memory)
def c3_dir():           return f"{RESULTS_ROOT}/results_c3_vggt{SUFFIX}"        # C3 Global (World Memory)

# ---- third-party model assets (set these to your local checkpoints/repos) ----
# SEA-RAFT optical-flow config json (https://github.com/princeton-vl/SEA-RAFT)
SEARAFT_CFG   = os.environ.get("SEARAFT_CFG", "")
# Depth-Anything-3 model id or local path (https://github.com/DepthAnything/Depth-Anything-3)
DA3_MODEL     = os.environ.get("DA3_MODEL", "depth-anything/DA3-LARGE-1.1")
# VGGT-Omega repo path for the feed-forward reconstruction (C3)
VGGT_OMEGA    = os.environ.get("VGGT_OMEGA_PATH", "")
# Optional local VGGT-Omega checkpoint file (empty -> let the repo resolve/download it)
VGGT_OMEGA_CKPT = os.environ.get("VGGT_OMEGA_CKPT", "")
# Q-Align / OneAlign model id (https://github.com/Q-Future/Q-Align)
QALIGN_MODEL  = os.environ.get("QALIGN_MODEL", "q-future/one-align")

def ensure_dirs():
    for d in (flow_dir(), flow_cache(), quality_dir(), consistency_dir(), c3_dir()):
        os.makedirs(d, exist_ok=True)
