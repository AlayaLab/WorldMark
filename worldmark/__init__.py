"""WorldMark: a reproducible benchmark for interactive world models.

Three metric categories (9 metrics total):
  - Action Dynamics : Direction Accuracy, Direction Purity, Motion Stability, Response Latency
  - World Memory    : Local Memory, Global Memory, Revisit Memory
  - Visual Quality  : Perceptual Quality, Aesthetic Quality

Stage scripts (run standalone or via ../run_eval.sh):
  flow_batch_full.py        -> Action Dynamics (M1-M4)
  qalign_batch.py           -> Visual Quality
  vggt_omega_global_c3.py   -> World Memory: Global
  consistency_v2.py         -> World Memory: Local (C1_ONLY=1) + Revisit (C2_ONLY=1)
  final_master.py / render_master_table.py -> aggregate + plot

All paths come from wm_config.py (environment-driven; no hardcoded absolute paths).
"""
__version__ = "0.1.0"
