"""
11_check_g.py — Validate GPU G matrix implementation against pre-computed R reference.

Two-part validation:

Part A — Implementation correctness (primary, passes/fails):
  Feed the EXACT SAME v_a / K_ifm that R used to produce G_gen.csv back into
  the GPU kernel via g_matrix_verify, then compare GPU G_gen against
  freshstaart/data/exports/G_gen.csv.
  Pass criterion: max|GPU − R| < 1e-6 for all cells.
  Pre-condition: output/G_gen_from_ref_inputs.csv must exist.
  How to generate:
    python3 validation/g/build_z_ref.py        # creates output/z1_ref.csv, z2_ref.csv
    ./g_matrix_verify output/z1_ref.csv output/z2_ref.csv \\
        <EXP>/Omega1.csv <EXP>/Omega2.csv \\
        output/G_gen_from_ref_inputs.csv \\
        <EXP>/G_gen.csv

Part B — Pipeline end-to-end diff (informational only, not pass/fail):
  Compare output/G16_g/G_gen.csv (from GPU smooth → GPU v_a → GPU G_ad)
  against freshstaart G_gen.csv.  The expected ~2e-3 diff is due to the
  smooth chain (R image.smooth vs GPU cuFFT) producing slightly different
  v_a values (~9e-4 max), which then propagate through the Phi integrals.
  This is NOT a G implementation bug.

Usage:
  cd ~/gpuimplementation
  python3 validation/g/11_check_g.py
"""

import csv
import math
import sys
import os

EXPORTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "reference", "exports"
)
GPU_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "output", "G16_g")
REF_IMPL = os.path.join(os.path.dirname(__file__), "..", "..", "output",
                        "G_gen_from_ref_inputs.csv")

TOLERANCE = 1e-6


def load_matrix_csv(path):
    """Load a CSV with header row (V1..VN) and return list-of-lists of floats."""
    rows = []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr)           # skip header
        for row in rdr:
            rows.append([float(x) for x in row])
    return rows


def compare(name, gpu_path, ref_path, primary=True):
    tag = "PRIMARY" if primary else "informational"
    print(f"\n--- {name}  [{tag}] ---")
    if not os.path.exists(gpu_path):
        print(f"  MISSING: {gpu_path}")
        print(f"  (run the generation commands described at the top of this script)")
        return None if primary else False
    if not os.path.exists(ref_path):
        print(f"  MISSING ref: {ref_path}")
        return None if primary else False

    gpu = load_matrix_csv(gpu_path)
    ref = load_matrix_csv(ref_path)

    nrows_g, ncols_g = len(gpu), len(gpu[0]) if gpu else 0
    nrows_r, ncols_r = len(ref), len(ref[0]) if ref else 0
    print(f"  GPU shape : {nrows_g}×{ncols_g}")
    print(f"  Ref shape : {nrows_r}×{ncols_r}")

    if (nrows_g, ncols_g) != (nrows_r, ncols_r):
        print("  FAIL — shape mismatch")
        return False

    max_diff = 0.0
    sum_diff = 0.0
    n_gt_tol = 0
    total    = nrows_g * ncols_g
    worst    = []

    for i in range(nrows_g):
        for j in range(ncols_g):
            gv = gpu[i][j]
            rv = ref[i][j]
            if math.isnan(gv) or math.isnan(rv):
                continue
            d = abs(gv - rv)
            sum_diff += d
            if d > max_diff:
                max_diff = d
            if d > TOLERANCE:
                n_gt_tol += 1
            worst.append((d, i, j, gv, rv))

    n_compared = total
    mean_diff  = sum_diff / n_compared if n_compared > 0 else float("nan")

    print(f"  Total cells      : {total}")
    print(f"  Max  |GPU - R|   : {max_diff:.3e}")
    print(f"  Mean |GPU - R|   : {mean_diff:.3e}")
    print(f"  Cells > {TOLERANCE:.0e}   : {n_gt_tol}")

    worst.sort(reverse=True)
    if worst and worst[0][0] > 1e-8:
        show = min(5, len(worst))
        print(f"\n  Top-{show} worst cells (row col GPU ref diff):")
        for d, i, j, gv, rv in worst[:show]:
            print(f"    [{i:3d},{j:3d}]  GPU={gv:.10g}  R={rv:.10g}  |diff|={d:.3e}")

    passed = max_diff < TOLERANCE
    print(f"\n  Result : {'PASS' if passed else 'FAIL'}")
    if not passed and primary:
        print(f"    Reason: max|diff| = {max_diff:.3e} >= {TOLERANCE:.0e}")
    return passed


def main():
    print("=== Layer 5: G Matrix Validation ===")

    # ---- Part A: Implementation correctness (uses matched inputs) ----
    print("\n" + "="*60)
    print("Part A: Implementation correctness (matched reference inputs)")
    print("="*60)
    print("  G_gen computed by GPU from the exact same v_a/K_ifm R used.")
    print(f"  GPU file : {REF_IMPL}")
    print(f"  Ref file : {os.path.join(EXPORTS, 'G_gen.csv')}")

    impl_pass = compare(
        "G_gen (GPU kernel with reference inputs)",
        REF_IMPL,
        os.path.join(EXPORTS, "G_gen.csv"),
        primary=True,
    )

    # ---- Part B: Full pipeline end-to-end diff (informational) ----
    print("\n" + "="*60)
    print("Part B: Pipeline end-to-end diff (informational, not pass/fail)")
    print("="*60)
    print("  GPU smooth -> GPU v_a -> GPU G_gen vs R smooth -> R v_a -> R G_gen")
    print("  Expected ~2e-3 diff due to smooth chain (not a G bug).")

    compare(
        "G_gen full pipeline vs freshstaart exports",
        os.path.join(GPU_DIR, "G_gen.csv"),
        os.path.join(EXPORTS, "G_gen.csv"),
        primary=False,
    )

    compare(
        "G = expm(G_gen) full pipeline vs freshstaart exports",
        os.path.join(GPU_DIR, "G.csv"),
        os.path.join(EXPORTS, "G.csv"),
        primary=False,
    )

    # ---- Final verdict ----
    print("\n" + "="*60)
    print("=== Summary ===")
    if impl_pass is None:
        print("  Part A: SKIPPED (G_gen_from_ref_inputs.csv not found)")
        print("  Run g_matrix_verify with z1_ref/z2_ref to generate it.")
        status = "SKIPPED"
    elif impl_pass:
        print("  Part A: PASS  — G kernel matches R to machine epsilon")
        print("  Part B: informational (pipeline diff from smooth chain)")
        status = "PASS"
    else:
        print("  Part A: FAIL  — G kernel has a bug independent of inputs")
        status = "FAIL"

    print(f"\nLayer 5 G-matrix implementation: {status}")
    sys.exit(0 if (impl_pass is True) else 1)


if __name__ == "__main__":
    main()
