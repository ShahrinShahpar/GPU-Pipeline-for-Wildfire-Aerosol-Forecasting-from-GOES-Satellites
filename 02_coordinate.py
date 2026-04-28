"""
Generate lat/lon coordinate lookup tables for GOES-16 and GOES-17.

This is the author's exact coordinate generation logic (Guanzhou Wei).
It reads the satellite projection parameters from one downloaded AOD NC file
per satellite and converts the 1500x2500 scan-angle grid into geographic
latitude/longitude coordinates.

Output files (saved to data/AOD/):
    g16_lats.txt  --  1500 x 2500 latitude  matrix for GOES-16 scan pixels
    g16_lons.txt  --  1500 x 2500 longitude matrix for GOES-16 scan pixels
    g17_lats.txt  --  1500 x 2500 latitude  matrix for GOES-17 scan pixels
    g17_lons.txt  --  1500 x 2500 longitude matrix for GOES-17 scan pixels

These files are static -- they depend only on the satellite's fixed orbital
position, not on the date or time. Run this once after downloading the NC files.
"""

from netCDF4 import Dataset
from pyproj import Proj
import numpy as np
import glob
import os

OUT_DIR = os.path.join("data", "AOD")

for satellite, sat_num in [("GOES-16", "16"), ("GOES-17", "17")]:
    # Pick the first available NC file for this satellite
    pattern = os.path.join("data", "AOD", satellite, "275", "18", "*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No NC files found for {satellite} at {pattern}")
        print("       Download the data first (Step 1), then re-run this script.")
        continue

    nc_path = files[0]
    print(f"\n{satellite}: using {os.path.basename(nc_path)}")

    # Read satellite projection parameters (author's exact logic)
    file = Dataset(nc_path)
    sat_h     = file.variables['goes_imager_projection'].perspective_point_height
    sat_lon   = file.variables['goes_imager_projection'].longitude_of_projection_origin
    sat_sweep = file.variables['goes_imager_projection'].sweep_angle_axis
    X = file.variables['x'][:] * sat_h
    Y = file.variables['y'][:] * sat_h
    file.close()

    print(f"  sat_h={sat_h:.0f}  sat_lon={sat_lon}  sweep={sat_sweep}")

    # Project scan angles to lat/lon (spherical Earth, author's exact parameters)
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep, a=6378137.0)
    XX, YY = np.meshgrid(X, Y)
    lons, lats = p(XX, YY, inverse=True)

    # Mask pixels outside the satellite's field of view
    mask = (lons == lons[0][0])
    lons[mask] = np.nan
    lats[mask] = np.nan

    # Save with 2-decimal precision (author's exact format)
    lat_out = os.path.join(OUT_DIR, f"g{sat_num}_lats.txt")
    lon_out = os.path.join(OUT_DIR, f"g{sat_num}_lons.txt")
    print(f"  Saving {lat_out} ...")
    np.savetxt(lat_out, lats, fmt='%.2f')
    print(f"  Saving {lon_out} ...")
    np.savetxt(lon_out, lons, fmt='%.2f')
    print(f"  Done. Grid shape: {lats.shape}")

print("\nCoordinate files written to data/AOD/")
