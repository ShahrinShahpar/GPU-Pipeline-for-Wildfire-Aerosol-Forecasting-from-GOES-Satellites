"""
03_preprocess.py
----------------
Convert GOES-16 AND GOES-17 NetCDF AOD files + coordinate txt files into
per-image binary files that regrid_gpu can read.

Binary format per image file (little-endian):
  int32      n    -- number of filtered points
  float64[n] lats -- latitudes  (filtered to region, NaN removed)
  float64[n] lons -- longitudes
  float64[n] aods -- AOD values (clamped >= 0)

Output layout:
  bin_images/G16/img_001.bin ... img_060.bin
  bin_images/G17/img_001.bin ... img_060.bin
  manifest_G16.txt
  manifest_G17.txt

Run from inside gpuimplementation/:
  python 03_preprocess.py
"""

import os
import sys
import struct
import time
import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    sys.exit("netCDF4 Python package not found. pip install netCDF4")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# All data lives under gpuimplementation/data/AOD/
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "AOD")

# Spatial domain (matching the paper)
LAT_MIN, LAT_MAX = 35.0, 37.4
LON_MIN, LON_MAX = -124.0, -121.6

# Hours to process
HOURS              = list(range(18, 23))
MAX_FILES_PER_HOUR = 12

SATELLITES = [
    {
        "name":     "G16",
        "sat_dir":  os.path.join(DATA_DIR, "GOES-16", "275"),
        "lat_file": os.path.join(DATA_DIR, "g16_lats.txt"),
        "lon_file": os.path.join(DATA_DIR, "g16_lons.txt"),
    },
    {
        "name":     "G17",
        "sat_dir":  os.path.join(DATA_DIR, "GOES-17", "275"),
        "lat_file": os.path.join(DATA_DIR, "g17_lats.txt"),
        "lon_file": os.path.join(DATA_DIR, "g17_lons.txt"),
    },
]

_t_total_start = time.perf_counter()
timing_per_sat = {}


def process_satellite(sat):
    name     = sat["name"]
    sat_dir  = sat["sat_dir"]
    lat_file = sat["lat_file"]
    lon_file = sat["lon_file"]
    out_dir  = os.path.join(SCRIPT_DIR, "bin_images", name)
    manifest = os.path.join(SCRIPT_DIR, f"manifest_{name}.txt")

    print(f"\n{'='*60}")
    print(f"  Satellite: {name}")
    print(f"{'='*60}")

    if not os.path.isdir(sat_dir):
        print(f"  WARNING: satellite directory not found: {sat_dir}  — skipping")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    # 1. Load coordinate matrices
    print(f"  Loading coordinate files ...")
    _t0 = time.perf_counter()
    lat_arr = np.loadtxt(lat_file, dtype=np.float64)
    lon_arr = np.loadtxt(lon_file, dtype=np.float64)
    print(f"    Shape: {lat_arr.shape}  ({time.perf_counter()-_t0:.2f}s)")

    if lat_arr.shape != lon_arr.shape:
        print(f"  ERROR: lat/lon shape mismatch — skipping {name}")
        return 0

    # 2. Collect file paths
    print(f"  Scanning file paths ...")
    _t0 = time.perf_counter()
    file_paths = {}
    for hour in HOURS:
        hour_dir = os.path.join(sat_dir, str(hour))
        if not os.path.isdir(hour_dir):
            continue
        files = sorted(f for f in os.listdir(hour_dir) if not f.startswith("."))
        for j, fname in enumerate(files[:MAX_FILES_PER_HOUR]):
            file_paths[(hour, j)] = os.path.join(hour_dir, fname)
        print(f"    Hour {hour}: {min(len(files), MAX_FILES_PER_HOUR)} files")
    print(f"    Scan done ({time.perf_counter()-_t0:.2f}s)")

    # 3. Process each image
    manifest_lines = []
    _t_proc = time.perf_counter()

    for i, hour in enumerate(HOURS):
        for j in range(MAX_FILES_PER_HOUR):
            img_index_1based = i * MAX_FILES_PER_HOUR + (j + 1)
            fp = file_paths.get((hour, j))
            if fp is None:
                continue

            print(f"  [{img_index_1based:02d}/60] {os.path.basename(fp)}", end="", flush=True)

            try:
                nc = Dataset(fp, "r")
            except Exception as e:
                print(f"  SKIP (cannot open): {e}")
                continue

            if "AOD" not in nc.variables:
                nc.close()
                print("  SKIP (no AOD variable)")
                continue

            aod_var = nc.variables["AOD"]
            nc.set_auto_maskandscale(True)
            aod_raw = aod_var[:]

            if isinstance(aod_raw, np.ma.MaskedArray):
                aod_data = np.asarray(aod_raw.filled(np.nan), dtype=np.float64)
            else:
                aod_data = np.asarray(aod_raw, dtype=np.float64)
                fill_val = getattr(aod_var, "_FillValue", None)
                if fill_val is not None:
                    aod_data[np.isclose(aod_data, float(fill_val), rtol=1e-5)] = np.nan

            nc.close()

            # Align AOD shape with coordinate matrices (handle x/y dim ordering)
            if aod_data.shape == lat_arr.shape:
                pass
            elif aod_data.shape == (lat_arr.shape[1], lat_arr.shape[0]):
                aod_data = aod_data.T
            else:
                print(f"  SKIP: AOD shape {aod_data.shape} incompatible "
                      f"with coord shape {lat_arr.shape}")
                continue

            # Filter: drop NaN coords + NaN AOD + outside spatial domain
            # Uses strict inequalities (>) matching the R Regrid function's subset()
            valid = (
                np.isfinite(lat_arr) & np.isfinite(lon_arr) & np.isfinite(aod_data) &
                (lat_arr > LAT_MIN) & (lat_arr < LAT_MAX) &
                (lon_arr > LON_MIN) & (lon_arr < LON_MAX)
            )

            # Keep as float64 throughout — boundary comparisons in the GPU kernel
            # must match R's float64 arithmetic exactly.
            lats_f = lat_arr[valid]
            lons_f = lon_arr[valid]
            aods_f = np.clip(aod_data[valid], 0.0, None)

            n = len(lats_f)
            print(f"  -> {n} pts")

            if n == 0:
                continue

            out_path = os.path.join(out_dir, f"img_{img_index_1based:03d}.bin")
            with open(out_path, "wb") as f:
                f.write(struct.pack("<i", n))
                f.write(lats_f.tobytes())
                f.write(lons_f.tobytes())
                f.write(aods_f.tobytes())

            manifest_lines.append(f"{img_index_1based}\t{out_path}")

    # 4. Write manifest
    with open(manifest, "w") as f:
        f.write("\n".join(manifest_lines) + "\n")

    _t_elapsed = time.perf_counter() - _t_proc
    timing_per_sat[name] = _t_elapsed

    print(f"\n  {name} done: {len(manifest_lines)} images in {_t_elapsed:.2f}s "
          f"({_t_elapsed/max(len(manifest_lines),1):.3f}s/image)")
    print(f"  Manifest: {manifest}")
    return len(manifest_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
total_images = 0
for sat in SATELLITES:
    total_images += process_satellite(sat)

_t_total = time.perf_counter() - _t_total_start

print(f"\n{'='*60}")
print(f"PREPROCESSING COMPLETE")
print(f"{'='*60}")
for sat_name, t in timing_per_sat.items():
    print(f"  {sat_name}: {t:.2f} s")
print(f"  Total wall time : {_t_total:.2f} s")
print(f"  Total images    : {total_images}")
print(f"{'='*60}")
print(f"\nNext step:  make && ./regrid_gpu manifest_G16.txt output/G16 && ./regrid_gpu manifest_G17.txt output/G17")
