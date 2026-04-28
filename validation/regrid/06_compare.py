"""
06_compare.py
-------------
Compare GPU-regridded output (output/G16_processed.csv, output/G17_processed.csv)
against the bundled reference (reference/regrid_output/G16_processed.csv, etc.)

Reports per time-step and overall:
  - Pearson correlation
  - RMSE
  - MAE
  - Mean AOD
  - NaN fraction

Run from inside gpuimplementation/:
  python 06_compare.py

Pass --ref <path> to override the reference directory:
  python 06_compare.py --ref /some/other/path

Reference data is bundled in reference/regrid_output/ inside this repository.
"""

import os
import sys
import math
import csv
import argparse

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = os.path.join(SCRIPT_DIR, "..", "..", "reference", "regrid_output")
N_GRID      = 3600
N_IMAGES    = 60


def parse_csv(path):
    """Return dict: time_index -> list of AOD values (float or nan)."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["time_index"])
            v = row["AOD"].strip()
            aod = float("nan") if v in ("NaN", "nan", "NA", "") else float(v)
            data.setdefault(t, []).append(aod)
    return data


def pearson(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b)


def rmse(a, b):
    if not a:
        return float("nan")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def mae(a, b):
    if not a:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def compare_satellite(sat_name, ref_dir):
    gpu_path = os.path.join(SCRIPT_DIR, "output", f"{sat_name}_processed.csv")
    ref_path = os.path.join(ref_dir, f"{sat_name}_processed.csv")

    print(f"\n{'='*68}")
    print(f"  {sat_name}")
    print(f"  GPU : {gpu_path}")
    print(f"  REF : {ref_path}")
    print(f"{'='*68}")

    for p, label in [(gpu_path, "GPU"), (ref_path, "REF")]:
        if not os.path.exists(p):
            print(f"  ERROR: {label} file not found: {p}")
            return

    gpu_data = parse_csv(gpu_path)
    ref_data = parse_csv(ref_path)

    print(f"\n  {'t':>4}  {'N_valid':>7}  {'Corr':>7}  {'RMSE':>8}  {'MAE':>8}  "
          f"{'Mean_GPU':>9}  {'Mean_REF':>9}  {'NaN_GPU%':>8}  {'NaN_REF%':>8}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  "
          f"{'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}")

    all_gpu, all_ref = [], []
    all_corrs = []

    for t in range(1, N_IMAGES + 1):
        g = gpu_data.get(t, [float("nan")] * N_GRID)
        r = ref_data.get(t, [float("nan")] * N_GRID)

        nan_g = sum(1 for v in g if math.isnan(v))
        nan_r = sum(1 for v in r if math.isnan(v))

        # Only compare cells where both are valid
        valid_pairs = [(gv, rv) for gv, rv in zip(g, r)
                       if not math.isnan(gv) and not math.isnan(rv)]
        gv_vals = [p[0] for p in valid_pairs]
        rv_vals = [p[1] for p in valid_pairs]

        corr = pearson(gv_vals, rv_vals)
        r_val = rmse(gv_vals, rv_vals)
        m_val = mae(gv_vals, rv_vals)
        mean_g = sum(gv_vals) / len(gv_vals) if gv_vals else float("nan")
        mean_r = sum(rv_vals) / len(rv_vals) if rv_vals else float("nan")

        print(f"  {t:>4}  {len(valid_pairs):>7}  {corr:>7.4f}  {r_val:>8.5f}  {m_val:>8.5f}  "
              f"{mean_g:>9.5f}  {mean_r:>9.5f}  "
              f"{100*nan_g/len(g):>7.1f}%  {100*nan_r/len(r):>7.1f}%")

        all_gpu.extend(gv_vals)
        all_ref.extend(rv_vals)
        if not math.isnan(corr):
            all_corrs.append(corr)

    print(f"\n  {'OVERALL':->60}")
    overall_corr = pearson(all_gpu, all_ref)
    overall_rmse = rmse(all_gpu, all_ref)
    overall_mae  = mae(all_gpu, all_ref)
    mean_all_g   = sum(all_gpu) / len(all_gpu) if all_gpu else float("nan")
    mean_all_r   = sum(all_ref) / len(all_ref) if all_ref else float("nan")
    mean_corr    = sum(all_corrs) / len(all_corrs) if all_corrs else float("nan")

    print(f"  Overall correlation  : {overall_corr:.6f}")
    print(f"  Mean per-step corr   : {mean_corr:.6f}")
    print(f"  Overall RMSE         : {overall_rmse:.6f}")
    print(f"  Overall MAE          : {overall_mae:.6f}")
    print(f"  Mean AOD  (GPU)      : {mean_all_g:.6f}")
    print(f"  Mean AOD  (REF)      : {mean_all_r:.6f}")

    if overall_corr >= 0.99:
        print(f"\n  PASS — correlation >= 0.99, GPU regrid matches reference.")
    elif overall_corr >= 0.85:
        print(f"\n  CLOSE — correlation >= 0.85 (archive drift expected, see freshstaart README).")
    else:
        print(f"\n  WARN — correlation < 0.85, check preprocessing steps.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ref", default=DEFAULT_REF,
                    help="Path to reference output directory (default: reference/regrid_output)")
args = parser.parse_args()

print("="*68)
print("STEP 6: Comparing GPU output against bundled reference")
print(f"Reference directory: {args.ref}")
print("="*68)

compare_satellite("G16", args.ref)
compare_satellite("G17", args.ref)

print("\nDone.")
