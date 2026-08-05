WorldMark benchmark inputs: first-/third-person starting images + action sequences.
  first_view/  real|style  : 25 first-person starting images (000..024.jpg) + *_action.txt + *_intrinsics
  third_view/  real|style  : 25 third-person starting images + *_action.txt + *_intrinsics + prompt_*.txt
  action_protocol.txt      : action_id -> key sequence
The per-image action assignment (*_action.txt) is identical for first and third view.
Each (image, action) pair is one generated video, named {image}_{action:03d}.mp4.
