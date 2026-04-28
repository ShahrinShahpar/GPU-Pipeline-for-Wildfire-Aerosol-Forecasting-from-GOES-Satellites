#!/usr/bin/env python3
"""
prep_verify_input.py — Convert a GPU regrid CSV to 60x60 R-matrix format
for feeding into knn_impute_verify.

The GPU regrid CSV (output/G16/img_NNN.csv) has columns: long, lat, AOD
with 3600 rows ordered g = lat_i*60 + lon_i (lon varies fastest).

knn_impute_verify expects a 60x60 matrix with header V1..V60 where:
  row r (0-indexed) = all latitudes at longitude r
  col c (0-indexed) = latitude c
This is the same layout as R's matrix(aod_vec, 60, 60) column-major fill.

Usage (from gpuimplementation/ root):
  python validation/knn/prep_verify_input.py STEP [OUTPUT_CSV]

  STEP       — time step number (1..60)
  OUTPUT_CSV — output path (default: /tmp/knn_input_STEP.csv)

Example:
  python validation/knn/prep_verify_input.py 3
  ./knn_impute_verify /tmp/knn_input_003.csv \\
      validation/knn/ref_G16/img_003.csv \\
      /tmp/knn_output_003.csv
"""

import sys
import os
import numpy as np
import pandas as pd

NR = 60

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    step = int(sys.argv[1])
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/knn_input_{step:03d}.csv"

    in_path = os.path.join("output", "G16", f"img_{step:03d}.csv")
    if not os.path.isfile(in_path):
        print(f"ERROR: regrid CSV not found: {in_path}")
        sys.exit(1)

    df = pd.read_csv(in_path, header=0, na_values=["NaN", "nan", "NA"])
    if len(df) != NR * NR:
        print(f"ERROR: expected {NR*NR} rows, got {len(df)}")
        sys.exit(1)

    # AOD vector in g = lat_i*60 + lon_i order
    aod = df["AOD"].values.astype(np.float64)

    # Reshape to (lat, lon) then transpose to (lon, lat) — same as R's
    # matrix(aod_vec, Nr, Nr) column-major: mat[lon_i, lat_i] = aod[lat_i*Nr + lon_i]
    mat_lat_lon = aod.reshape(NR, NR)      # shape (lat, lon)
    mat_lon_lat = mat_lat_lon.T             # shape (lon, lat)

    # Write with header V1..V60 and full float64 precision
    header = ",".join(f"V{j+1}" for j in range(NR))
    lines = [header]
    for i in range(NR):
        row_parts = []
        for j in range(NR):
            v = mat_lon_lat[i, j]
            if np.isnan(v):
                row_parts.append("NaN")
            else:
                row_parts.append(f"{v:.17g}")
        lines.append(",".join(row_parts))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    nan_count = int(np.isnan(mat_lon_lat).sum())
    print(f"Written: {out_path}  ({NR}x{NR}, {nan_count} NaN cells)")
    print(f"\nRun:")
    print(f"  ./knn_impute_verify {out_path} \\")
    print(f"      validation/knn/ref_G16/img_{step:03d}.csv \\")
    print(f"      /tmp/knn_output_{step:03d}.csv")

if __name__ == "__main__":
    main()
