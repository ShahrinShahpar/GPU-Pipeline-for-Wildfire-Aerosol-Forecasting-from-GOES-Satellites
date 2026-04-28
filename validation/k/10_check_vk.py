#!/usr/bin/env python3
"""
10_check_vk.py  — Layer 4: K diffusivity validation

Compares GPU-computed velocity (v_a) and diffusivity (K_ifm) against
R reference files from application.R.

Usage:
    python3 validation/k/10_check_vk.py

Pass criterion: max|GPU - R| < 1e-6 for all fields.
"""

import numpy as np
import sys
import os

# ── paths ────────────────────────────────────────────────────────────────────
GPU_VA    = "output/G16_k/v_a.csv"
GPU_KIFM  = "output/G16_k/K_ifm.csv"
REF_VA    = "validation/k/ref_G16/v_a.csv"
REF_KIFM  = "validation/k/ref_G16/K_ifm.csv"

TOL = 1e-6

def load_csv(path):
    """Load a CSV and return (header_list, data as float64 ndarray)."""
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None, None
    import csv
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append([float(x) if x not in ("NA", "NaN", "nan", "") else np.nan
                         for x in row])
    return header, np.array(rows, dtype=np.float64)

def check_field(label, gpu_col, ref_col):
    """Compare two 1-D arrays; print statistics; return True if pass."""
    mask = ~(np.isnan(gpu_col) | np.isnan(ref_col))
    nan_gpu = np.sum(np.isnan(gpu_col))
    nan_ref = np.sum(np.isnan(ref_col))
    nan_mismatch = int(np.sum(np.isnan(gpu_col) != np.isnan(ref_col)))

    diff = np.abs(gpu_col[mask] - ref_col[mask])
    max_diff = float(diff.max()) if len(diff) > 0 else 0.0
    mean_diff = float(diff.mean()) if len(diff) > 0 else 0.0

    ok = max_diff < TOL and nan_mismatch == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label:12s}  max|diff|={max_diff:.3e}  "
          f"mean|diff|={mean_diff:.3e}  "
          f"NaN gpu/ref/mismatch={nan_gpu}/{nan_ref}/{nan_mismatch}")
    return ok

# ── v_a (long, lat, v_x, v_y) ────────────────────────────────────────────────
print("=== Layer 4: velocity (v_a) ===")
_, gpu_va = load_csv(GPU_VA)
_, ref_va = load_csv(REF_VA)

all_ok = True
if gpu_va is None or ref_va is None:
    print("  Cannot compare — file missing.")
    all_ok = False
elif gpu_va.shape != ref_va.shape:
    print(f"  Shape mismatch: GPU {gpu_va.shape} vs R {ref_va.shape}")
    all_ok = False
else:
    print(f"  Rows: {gpu_va.shape[0]}   Cols: GPU={gpu_va.shape[1]}  R={ref_va.shape[1]}")
    all_ok &= check_field("v_x", gpu_va[:, 2], ref_va[:, 2])
    all_ok &= check_field("v_y", gpu_va[:, 3], ref_va[:, 3])

# ── K_ifm (long, lat, K, dK_dx, dK_dy) ──────────────────────────────────────
print("\n=== Layer 4: diffusivity K (K_ifm) ===")
_, gpu_k = load_csv(GPU_KIFM)
_, ref_k = load_csv(REF_KIFM)

if gpu_k is None or ref_k is None:
    print("  Cannot compare — file missing.")
    all_ok = False
elif gpu_k.shape != ref_k.shape:
    print(f"  Shape mismatch: GPU {gpu_k.shape} vs R {ref_k.shape}")
    all_ok = False
else:
    print(f"  Rows: {gpu_k.shape[0]}   Cols: GPU={gpu_k.shape[1]}  R={ref_k.shape[1]}")
    all_ok &= check_field("K",      gpu_k[:, 2], ref_k[:, 2])
    all_ok &= check_field("dK_dx",  gpu_k[:, 3], ref_k[:, 3])
    all_ok &= check_field("dK_dy",  gpu_k[:, 4], ref_k[:, 4])

# ── summary ───────────────────────────────────────────────────────────────────
print()
if all_ok:
    print("Layer 4 PASS  — all fields within tolerance 1e-6.")
else:
    print("Layer 4 FAIL  — see above for details.")
    sys.exit(1)
