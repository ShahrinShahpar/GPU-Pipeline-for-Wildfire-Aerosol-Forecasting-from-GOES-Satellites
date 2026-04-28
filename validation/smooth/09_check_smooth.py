#!/usr/bin/env python3
"""
09_check_smooth.py — Compare GPU image-smooth output vs R fields::image.smooth reference.

Run from gpuimplementation/ root:
    python validation/smooth/09_check_smooth.py

Inputs:
    output/G16_smooth/speed_smooth.csv     GPU smoothed speed  (pixels/step / Nr)
    output/G16_smooth/angle_smooth.csv     GPU smoothed angle  (radians)
    validation/smooth/ref_G16/speed_smooth.csv  R reference speed
    validation/smooth/ref_G16/angle_smooth.csv  R reference angle

Pass criterion: max|GPU - R| < 1e-6 for both fields.
"""

import os, sys
import numpy as np
import pandas as pd

GPU_SMOOTH_DIR = os.path.join("output", "G16_smooth")
REF_DIR        = os.path.join("validation", "smooth", "ref_G16")
NR             = 60
PASS_TOL       = 1e-6

def load_matrix(path):
    df = pd.read_csv(path, header=0)
    mat = df.values.astype(np.float64)
    if mat.shape != (NR, NR):
        raise ValueError(f"Expected ({NR},{NR}), got {mat.shape} in {path}")
    return mat

def check_field(name, gpu_path, ref_path):
    try:
        gpu = load_matrix(gpu_path)
        ref = load_matrix(ref_path)
    except Exception as e:
        print(f"  {name}: ERROR loading — {e}")
        return False

    nan_mm = int((np.isnan(gpu) != np.isnan(ref)).sum())
    both   = ~np.isnan(gpu) & ~np.isnan(ref)
    if both.sum() == 0:
        print(f"  {name}: no finite cells to compare")
        return False
    d = np.abs(gpu[both] - ref[both])
    max_d  = float(d.max())
    mean_d = float(d.mean())
    ok = (nan_mm == 0 and max_d < PASS_TOL)
    print(f"  {name:20s}  NaN_mm={nan_mm}  max|diff|={max_d:.3e}  "
          f"mean|diff|={mean_d:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok

print("=" * 70)
print("IMAGE SMOOTH VERIFICATION — GPU vs R fields::image.smooth")
print(f"GPU smooth dir : {GPU_SMOOTH_DIR}")
print(f"R ref dir      : {REF_DIR}")
print(f"Tolerance      : < {PASS_TOL:.0e}")
print("=" * 70 + "\n")

for d, label, hint in [
    (GPU_SMOOTH_DIR, "GPU smooth output", "Run: ./aod_pipeline"),
    (REF_DIR,        "R smooth reference","Run: Rscript r_reference/save_smooth_reference.R"),
]:
    if not os.path.isdir(d):
        print(f"ERROR: {label} dir not found: {d}\n  {hint}")
        sys.exit(1)

ok_sp = check_field(
    "speed_smooth (÷Nr)",
    os.path.join(GPU_SMOOTH_DIR, "speed_smooth.csv"),
    os.path.join(REF_DIR,        "speed_smooth.csv"))

ok_an = check_field(
    "angle_smooth (rad)",
    os.path.join(GPU_SMOOTH_DIR, "angle_smooth.csv"),
    os.path.join(REF_DIR,        "angle_smooth.csv"))

print("\n" + "=" * 70)
if ok_sp and ok_an:
    print("PASS — both fields within tolerance (max|diff| < 1e-6).")
else:
    print("FAIL — one or more fields exceeded tolerance.")
    sys.exit(1)
