WorldMark benchmark inputs  —  read-only, never edit.

Layout
  action_protocol.txt              action_id (1..15) -> key sequence
  {view}/                          view = first_view | third_view
    {domain}/NNN.jpg               domain = real | style; 25 images, 000..024
    {domain}_intrinsics/NNN_intrinsics.npy
    {domain}_action.txt            row i = image i's 5 assigned action_ids
    prompt_{domain}.txt            row i = image i's caption (25 rows)

All four {view}/prompt_{domain}.txt exist. Captions differ by view: every first_view caption
starts with "First-person view." and every third_view caption with "Third-person view." — the
prompt already carries the view signal, so never reuse a caption across views or domains.

Facts
  *_action.txt is identical between first_view and third_view for a given domain.
  Intrinsics are (4,) float32 [fx, fy, cx, cy], per-image and all distinct, in ORIGINAL-image
    pixel units (cx ~ (W-1)/2, cy ~ (H-1)/2) — transform them for your model's resolution,
    do not pass them through.
  Image sizes are not uniform: real spans 1024x768 .. 1920x1200 (AR 1.33-1.79);
    style is uniformly 1024x1024 (AR 1.00).
  Horizontal FOV spans 22-90 degrees (median ~66 for real, ~45 for style).

Scope
  Each (image, action) pair is one generated video, named {image:03d}_{action:03d}.mp4.
  One (view, domain) pair = 25 images x 5 actions = 125 videos; all four = 500.

See ../generation/README.md to produce videos, and
../.claude/skills/world-model-adapter/references/output_spec.md for the full I/O contract.
