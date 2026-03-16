"""
Export the trained CRNN model to TorchScript format for PyTorch Mobile.

This converts best_model.pt → models/model.ptl
The .ptl file is then copied into the Android app's assets folder.

Usage:
    python3 export_mobile_model.py

Output:
    models/model.ptl   — copy this file to android/app/src/main/assets/
"""

import os
import sys
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.crnn import CRNN
from config import CHECKPOINTS_DIR, MODELS_DIR, IMG_HEIGHT, IMG_WIDTH

def export():
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load trained weights
    print("Loading model...")
    model = CRNN()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Trace the model with a dummy input (1, 1, H, W)
    # torch.jit.trace records all operations for a concrete input shape
    print("Tracing model...")
    example_input = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Optimise for mobile (fuses ops, removes unused nodes)
    print("Optimising for mobile...")
    optimised = optimize_for_mobile(traced)

    # Save as .ptl (PyTorch Lite Interpreter format)
    output_path = os.path.join(MODELS_DIR, "model.ptl")
    optimised._save_for_lite_interpreter(output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nExported successfully!")
    print(f"  Path:  {output_path}")
    print(f"  Size:  {size_mb:.1f} MB")
    print(f"\nNext step:")
    print(f"  Copy models/model.ptl  →  android/app/src/main/assets/model.ptl")

if __name__ == "__main__":
    export()
