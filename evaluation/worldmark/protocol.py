"""Action protocol: key semantics, segment boundaries, probe routing, expected-flow templates.

Key semantics: W forward / S backward / A pan-left / D pan-right / L yaw-left / R yaw-right. 20s per key.
FPP expected-flow templates:
  W = radial expansion (about the FoE center), S = radial contraction
  A = uniform (+x) (camera moves left -> content flows right), D = uniform (-x)
  L = uniform (+x) (gaze turns left -> content flows right, same template as A), R = uniform (-x) (same template as D)
Each key -> (mode, sign): mode in {"radial","xaxis"}, sign in {+1,-1}.
"""

SEG_SECONDS = 20.0

ACTION_TABLE = {
    1: "W", 2: "S", 3: "A", 4: "D", 5: "R",
    6: "WS", 7: "LR", 8: "AD", 9: "WR", 10: "SL",
    11: "WSW", 12: "LRL", 13: "WRW", 14: "ADA", 15: "WRS",
}

# key -> (template mode, sign). A/L share a template, D/R share a template
# (the first flow-domain version does not distinguish translation from rotation).
KEY_TEMPLATE = {
    "W": ("radial", +1.0),
    "S": ("radial", -1.0),
    "A": ("xaxis", +1.0),
    "D": ("xaxis", -1.0),
    "L": ("xaxis", +1.0),
    "R": ("xaxis", -1.0),
}

# Probe routing: transition type
REVERSE_SAME_FAMILY = {6, 7, 8}      # WS/LR/AD opposing reversal pairs (core)
CROSS_FAMILY = {9, 10}               # WR/SL cross-family switches
DOUBLE_REVERSAL = {11, 12, 14}       # WSW/LRL/ADA double reversal (repeat consistency)
HEADING_COMP = {13, 15}              # WRW/WRS heading composition
GAIN_SYM_ACTIONS = {6, 8, 7}         # gain symmetry
ATOMIC = {1, 2, 3, 4, 5}             # atomic segments


def keys_of(action_id: int) -> list[str]:
    return list(ACTION_TABLE[action_id])


def segment_boundaries(action_id: int) -> list[float]:
    """Internal segment boundaries (seconds), excluding 0 and the end. 1 key=[], 2 keys=[20], 3 keys=[20,40]."""
    n = len(ACTION_TABLE[action_id])
    return [SEG_SECONDS * i for i in range(1, n)]


def segments(action_id: int):
    """Return [(seg_idx, key, t_start, t_end), ...]."""
    ks = keys_of(action_id)
    return [(i, k, i * SEG_SECONDS, (i + 1) * SEG_SECONDS) for i, k in enumerate(ks)]


def transition_type(action_id: int) -> str:
    if action_id in REVERSE_SAME_FAMILY:
        return "reverse_same_family"
    if action_id in CROSS_FAMILY:
        return "cross_family"
    if action_id in DOUBLE_REVERSAL:
        return "double_reversal"
    if action_id in HEADING_COMP:
        return "heading_composition"
    return "none"
