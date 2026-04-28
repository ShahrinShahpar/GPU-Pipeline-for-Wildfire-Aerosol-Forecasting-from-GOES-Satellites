"""
05_merge.py
-----------
Combine the 60 per-image CSVs produced by regrid_gpu into a single
G16_processed.csv and G17_processed.csv — matching the format of
freshstaart/output/G16_processed.csv exactly.

Output format:
  long,lat,AOD,time_index
  -123.98,35.02,0.12045...,1
  ...                           (60 * 3600 = 216000 rows per file)

Run from inside gpuimplementation/:
  python 05_merge.py

Check when done:
  wc -l output/G16_processed.csv   # should be 216001 (header + 216000 rows)
  wc -l output/G17_processed.csv   # same
"""

import os
import csv
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_IMAGES   = 60
N_GRID     = 3600   # 60 x 60

def merge_satellite(sat_name):
    img_dir  = os.path.join(SCRIPT_DIR, "output", sat_name)
    out_path = os.path.join(SCRIPT_DIR, "output", f"{sat_name}_processed.csv")

    print(f"\n{sat_name}: merging {N_IMAGES} images -> {out_path}")

    missing = []
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["long", "lat", "AOD", "time_index"])

        for t in range(1, N_IMAGES + 1):
            img_path = os.path.join(img_dir, f"img_{t:03d}.csv")
            if not os.path.exists(img_path):
                missing.append(t)
                print(f"  WARNING: missing {img_path} — skipping time_index {t}")
                continue

            with open(img_path, "r") as fin:
                reader = csv.reader(fin)
                next(reader)   # skip header (long,lat,AOD)
                rows_written = 0
                for row in reader:
                    writer.writerow([row[0], row[1], row[2], t])
                    rows_written += 1

            if rows_written != N_GRID:
                print(f"  WARNING: img_{t:03d}.csv has {rows_written} rows (expected {N_GRID})")

            if t % 12 == 0:
                print(f"  Processed {t}/{N_IMAGES} images ...")

    if missing:
        print(f"  {len(missing)} images were missing: {missing}")
    else:
        print(f"  All {N_IMAGES} images merged successfully.")

    # Quick sanity check
    with open(out_path) as f:
        n_lines = sum(1 for _ in f)
    expected = N_IMAGES * N_GRID + 1   # +1 for header
    status = "OK" if n_lines == expected else f"MISMATCH (expected {expected})"
    print(f"  Line count: {n_lines}  [{status}]")
    return n_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("="*60)
print("STEP 5: Merging per-image CSVs into combined output files")
print("="*60)

g16_lines = merge_satellite("G16")
g17_lines = merge_satellite("G17")

print(f"\n{'='*60}")
print("MERGE COMPLETE")
print(f"  output/G16_processed.csv : {g16_lines} lines")
print(f"  output/G17_processed.csv : {g17_lines} lines")
print(f"{'='*60}")
print("\nNext step:  python 06_compare.py")
