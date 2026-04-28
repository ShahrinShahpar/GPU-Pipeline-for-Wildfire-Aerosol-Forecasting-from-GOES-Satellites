#!/usr/bin/env python3
"""
07_check_knn.py — Compare GPU KNN output against R bnstruct::knn.impute reference.

Run from gpuimplementation/ root:
    python validation/knn/07_check_knn.py

Pass criterion: max|GPU - R| < 1e-6 for all 60 time steps.

Inputs:
    output/G16_knn/img_NNN.csv         — GPU pipeline output (SAVE_INTERMEDIATES)
    validation/knn/ref_G16/img_NNN.csv — R reference (from save_knn_reference.R)
    output/G16/img_NNN.csv             — GPU regrid output (pre-KNN, for NaN mask)

Both KNN CSVs are 60×60 matrices (lon rows, lat cols) with a header row.
The regrid CSVs have columns long, lat, AOD — used to identify which cells
were originally NaN before imputation.
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GPU_KNN_DIR   = os.path.join("output", "G16_knn")
REF_DIR       = os.path.join("validation", "knn", "ref_G16")
GPU_REGRID_DIR = os.path.join("output", "G16")   # pre-KNN regrid output
N_STEPS = 60
NR      = 60
PASS_TOL = 1e-6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_matrix(path):
    """Read a NR×NR CSV (with header) and return a (NR, NR) float64 ndarray."""
    df = pd.read_csv(path, header=0)
    mat = df.values.astype(np.float64)
    if mat.shape != (NR, NR):
        raise ValueError(f"Expected ({NR},{NR}), got {mat.shape} in {path}")
    return mat

def load_regrid_nan_mask(path):
    """
    Read GPU regrid CSV (long, lat, AOD) and return a (NR, NR) bool mask
    where True = this cell was NaN before KNN imputation.

    The regrid CSV has 3600 rows ordered g = lat_i*NR + lon_i (lon fastest).
    The KNN matrix has row = lon_i, col = lat_i (transposed).
    So nan_mask[lon_i, lat_i] = isnan(AOD[lat_i*NR + lon_i]).
    """
    df = pd.read_csv(path, header=0)
    if "AOD" not in df.columns:
        return None
    aod = df["AOD"].values.astype(np.float64)   # length NR*NR, order lat×lon
    if len(aod) != NR * NR:
        return None
    aod_grid = aod.reshape(NR, NR)              # aod_grid[lat_i, lon_i]
    nan_mask = np.isnan(aod_grid).T             # nan_mask[lon_i, lat_i] = True if NaN
    return nan_mask


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
print("=" * 70)
print("KNN IMPUTATION VERIFICATION — GPU vs R reference")
print(f"GPU KNN dir   : {GPU_KNN_DIR}")
print(f"R ref dir     : {REF_DIR}")
print(f"Regrid dir    : {GPU_REGRID_DIR}")
print(f"Steps         : {N_STEPS}")
print(f"Tol           : max|GPU-R| < {PASS_TOL:.0e}")
print("=" * 70)

# Check directories exist
for d, label, hint in [
    (GPU_KNN_DIR, "GPU KNN output", "Run: ./aod_pipeline (compiled with -DSAVE_INTERMEDIATES)"),
    (REF_DIR,     "R reference",   "Run: Rscript r_reference/save_knn_reference.R"),
]:
    if not os.path.isdir(d):
        print(f"\nERROR: {label} dir not found: {d}\n  {hint}")
        sys.exit(1)

# Per-step results
header = (f"{'t':>4}  {'NaN_mm':>6}  {'max|diff|':>12}  {'mean|diff|':>12}"
          f"  {'imp_diff':>8}  {'nonnan_diff':>11}  {'status':>6}")
print(f"\n{header}")
print("-" * len(header))

all_max_diff = []
all_failed   = []

for t in range(1, N_STEPS + 1):
    fname        = f"img_{t:03d}.csv"
    gpu_path     = os.path.join(GPU_KNN_DIR,    fname)
    ref_path     = os.path.join(REF_DIR,        fname)
    regrid_path  = os.path.join(GPU_REGRID_DIR, fname)

    if not os.path.isfile(gpu_path):
        print(f"  {t:>3}  SKIP (GPU KNN file missing)")
        all_failed.append(t); continue
    if not os.path.isfile(ref_path):
        print(f"  {t:>3}  SKIP (R ref file missing)")
        all_failed.append(t); continue

    try:
        gpu = load_matrix(gpu_path)
        ref = load_matrix(ref_path)
    except Exception as e:
        print(f"  {t:>3}  ERROR loading: {e}")
        all_failed.append(t); continue

    nan_mm = int((np.isnan(gpu) != np.isnan(ref)).sum())

    both_finite = ~np.isnan(gpu) & ~np.isnan(ref)
    if both_finite.sum() == 0:
        max_diff = mean_diff = float("nan")
    else:
        diff     = np.abs(gpu[both_finite] - ref[both_finite])
        max_diff  = float(diff.max())
        mean_diff = float(diff.mean())

    # Separate imputed-cell differences from originally-non-NaN differences
    nan_mask = load_regrid_nan_mask(regrid_path) if os.path.isfile(regrid_path) else None

    imp_max    = float("nan")   # max|diff| in originally-NaN cells
    nonnan_max = float("nan")   # max|diff| in originally-non-NaN cells

    if nan_mask is not None and both_finite.sum() > 0:
        full_diff = np.full((NR, NR), np.nan)
        full_diff[both_finite] = np.abs(gpu[both_finite] - ref[both_finite])

        imp_cells    = nan_mask & both_finite
        nonnan_cells = (~nan_mask) & both_finite
        imp_max    = float(np.nanmax(full_diff[imp_cells]))    if imp_cells.any()    else 0.0
        nonnan_max = float(np.nanmax(full_diff[nonnan_cells])) if nonnan_cells.any() else 0.0

    step_pass = (nan_mm == 0) and (np.isnan(max_diff) or max_diff < PASS_TOL)
    status    = "PASS" if step_pass else "FAIL"

    if not step_pass:
        all_failed.append(t)
    all_max_diff.append(max_diff if not np.isnan(max_diff) else 0.0)

    print(f"  {t:>3}  {nan_mm:>6}  {max_diff:>12.3e}  {mean_diff:>12.3e}"
          f"  {imp_max:>8.2e}  {nonnan_max:>11.2e}  {status:>6}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
n_compared = len(all_max_diff)
if n_compared > 0:
    overall_max  = max(all_max_diff)
    overall_mean = sum(all_max_diff) / n_compared
    print(f"Steps compared      : {n_compared} / {N_STEPS}")
    print(f"Steps failed        : {len(all_failed)}")
    print(f"Max  |GPU - R|      : {overall_max:.3e}  (across all steps & cells)")
    print(f"Mean |GPU - R|      : {overall_mean:.3e}  (per-step mean)")
    print()
    print("Column guide:")
    print("  imp_diff    = max|diff| in cells that were NaN before imputation")
    print("  nonnan_diff = max|diff| in cells that were valid before imputation")
    print("  If nonnan_diff > 0: data layout mismatch (not just imputation difference)")
    print("  If nonnan_diff = 0 and imp_diff > tol: neighbor tie-breaking differs from R")
else:
    print("No steps compared — check file paths above.")

print()
if len(all_failed) == 0:
    print("PASS — all steps within tolerance (max|diff| < 1e-6).")
else:
    print(f"FAIL — {len(all_failed)} step(s) failed: {all_failed}")
    sys.exit(1)
