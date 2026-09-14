"""A toy job showing the contract kgr expects from a training script:

  * write checkpoints into --out (kgr's {output_dir}),
  * if a checkpoint for this run is already there, continue from it --
    that is how `kgr push --resume-from` carries a run across the 9 h limit,
  * save often and atomically, so a run killed at the wall leaves a usable file.
"""
import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=1000)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
ckpt = out / f"{args.run_name}_state.json"

state = json.loads(ckpt.read_text()) if ckpt.exists() else {"step": 0, "loss": 1.0}
print(f"device={args.device}  starting at step {state['step']}", flush=True)

while state["step"] < args.steps:
    state["step"] += 1
    state["loss"] *= 0.999
    if state["step"] % 100 == 0:
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, ckpt)          # atomic: never a half-written checkpoint
        print(f"step {state['step']}  loss {state['loss']:.4f}", flush=True)
    time.sleep(0.001)

print("done", flush=True)
