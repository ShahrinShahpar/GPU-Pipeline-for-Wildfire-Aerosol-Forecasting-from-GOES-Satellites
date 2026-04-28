#!/usr/bin/env python3
"""
08_check_of.py — Compare GPU optical-flow output vs R SpatialVx::OF reference.

Run from gpuimplementation/ root:
    python validation/of/08_check_of.py

Inputs:
    output/G16_of/speed_NNN.csv         GPU OF speed  (pixels/step)
    output/G16_of/angle_NNN.csv         GPU OF angle  (degrees 0..360)
    validation/of/ref_G16/speed_NNN.csv R reference speed
    validation/of/ref_G16/angle_NNN.csv R reference angle

Pass criterion: max|GPU - R| < 1e-6 for both speed and angle,
across all interior (non-NaN) cells, all 20 pairs.
"""

import os, sys
import numpy as np
import pandas as pd

GPU_OF_DIR = os.path.join("output", "G16_of")
REF_DIR    = os.path.join("validation", "of", "ref_G16")
N_PAIRS    = 20
NR         = 60
PASS_TOL   = 1e-6

def load_matrix(path):
    df = pd.read_csv(path, header=0)
    mat = df.values.astype(np.float64)
    if mat.shape != (NR, NR):
        raise ValueError(f"Expected ({NR},{NR}), got {mat.shape} in {path}")
    return mat

print("=" * 70)
print("OPTICAL FLOW VERIFICATION — GPU vs R reference (SpatialVx::OF)")
print(f"GPU OF dir : {GPU_OF_DIR}")
print(f"R ref dir  : {REF_DIR}")
print(f"Pairs      : {N_PAIRS}")
print(f"Tolerance  : < {PASS_TOL:.0e}")
print("=" * 70)

for d, label, hint in [
    (GPU_OF_DIR, "GPU OF output",   "Run: ./aod_pipeline (compiled with -DSAVE_INTERMEDIATES)"),
    (REF_DIR,    "R OF reference",  "Run: Rscript r_reference/save_of_reference.R"),
]:
    if not os.path.isdir(d):
        print(f"\nERROR: {label} dir not found: {d}\n  {hint}")
        sys.exit(1)

header = f"{'pair':>5}  {'sp_NaN_mm':>9}  {'sp_max':>10}  {'sp_mean':>10}  {'an_NaN_mm':>9}  {'an_max':>10}  {'an_mean':>10}  {'status':>6}"
print(f"\n{header}")
print("-" * len(header))

failed = []

for i in range(1, N_PAIRS + 1):
    fname_sp = f"speed_{i:03d}.csv"
    fname_an = f"angle_{i:03d}.csv"

    try:
        gpu_sp = load_matrix(os.path.join(GPU_OF_DIR, fname_sp))
        gpu_an = load_matrix(os.path.join(GPU_OF_DIR, fname_an))
        ref_sp = load_matrix(os.path.join(REF_DIR,    fname_sp))
        ref_an = load_matrix(os.path.join(REF_DIR,    fname_an))
    except Exception as e:
        print(f"  {i:>3}  ERROR: {e}")
        failed.append(i); continue

    def stats(gpu, ref):
        nan_mm = int((np.isnan(gpu) != np.isnan(ref)).sum())
        both = ~np.isnan(gpu) & ~np.isnan(ref)
        if both.sum() == 0:
            return nan_mm, float("nan"), float("nan")
        d = np.abs(gpu[both] - ref[both])
        return nan_mm, float(d.max()), float(d.mean())

    sp_nm, sp_max, sp_mean = stats(gpu_sp, ref_sp)
    an_nm, an_max, an_mean = stats(gpu_an, ref_an)

    ok = (sp_nm == 0 and an_nm == 0 and
          (np.isnan(sp_max) or sp_max < PASS_TOL) and
          (np.isnan(an_max) or an_max < PASS_TOL))
    if not ok:
        failed.append(i)

    status = "PASS" if ok else "FAIL"
    print(f"  {i:>3}  {sp_nm:>9}  {sp_max:>10.3e}  {sp_mean:>10.3e}"
          f"  {an_nm:>9}  {an_max:>10.3e}  {an_mean:>10.3e}  {status:>6}")

print("\n" + "=" * 70)
n_ok = N_PAIRS - len(failed)
print(f"Pairs compared  : {N_PAIRS}")
print(f"Pairs passed    : {n_ok}")
print(f"Pairs failed    : {len(failed)}")
if failed:
    print(f"Failed pairs    : {failed}")
print()
if not failed:
    print("PASS — all pairs within tolerance.")
else:
    print(f"FAIL — {len(failed)} pair(s) failed.")
    sys.exit(1)
