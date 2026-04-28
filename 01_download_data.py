"""
Step 1: Download GOES-16 and GOES-17 ABI-L2-AODC NetCDF files.

Date   : October 1, 2020  (day 275 of 2020)
Hours  : 18, 19, 20, 21, 22 UTC
Target : 12 files per hour x 5 hours x 2 satellites = 120 files total

Data source: NOAA public AWS S3 buckets (no credentials required)
  s3://noaa-goes16/ABI-L2-AODC/2020/275/{hour}/
  s3://noaa-goes17/ABI-L2-AODC/2020/275/{hour}/
"""

import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import os
import sys

YEAR = 2020
DOY  = 275
HOURS = [18, 19, 20, 21, 22]

SATELLITES = {
    "GOES-16": "noaa-goes16",
    "GOES-17": "noaa-goes17",
}

S3_BASE = "https://{bucket}.s3.amazonaws.com"
NS = "http://s3.amazonaws.com/doc/2006-03-01/"   # S3 XML namespace


def list_bucket(bucket, prefix):
    """Return list of object keys under prefix via S3 XML API."""
    keys = []
    continuation = None
    while True:
        url = f"{S3_BASE.format(bucket=bucket)}/?list-type=2&prefix={prefix}&delimiter=/"
        if continuation:
            url += f"&continuation-token={urllib.parse.quote(continuation)}"
        with urllib.request.urlopen(url, timeout=30) as r:
            tree = ET.parse(r)
        root = tree.getroot()
        for content in root.findall(f"{{{NS}}}Contents"):
            key = content.find(f"{{{NS}}}Key").text
            if key.endswith(".nc"):
                keys.append(key)
        is_truncated = root.find(f"{{{NS}}}IsTruncated")
        if is_truncated is not None and is_truncated.text.lower() == "true":
            continuation = root.find(f"{{{NS}}}NextContinuationToken").text
        else:
            break
    return keys


def download_file(bucket, key, dest_path):
    url = f"{S3_BASE.format(bucket=bucket)}/{key}"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"    ERROR downloading {os.path.basename(dest_path)}: {e}")
        return False


import urllib.parse   # needed for continuation token quoting

total_downloaded = 0
total_skipped    = 0
total_errors     = 0

for sat_name, bucket in SATELLITES.items():
    print(f"\n{'='*60}")
    print(f"  {sat_name}  (bucket: {bucket})")
    print(f"{'='*60}")

    for hour in HOURS:
        prefix = f"ABI-L2-AODC/{YEAR}/{DOY}/{hour:02d}/"
        out_dir = os.path.join("data", "AOD", sat_name, str(DOY), str(hour))

        print(f"\n  Hour {hour:02d} UTC  ->  {out_dir}")

        # List available files
        try:
            keys = list_bucket(bucket, prefix)
        except Exception as e:
            print(f"    ERROR listing bucket: {e}")
            continue

        print(f"    Found {len(keys)} files on S3")

        for key in sorted(keys):
            fname     = os.path.basename(key)
            dest_path = os.path.join(out_dir, fname)

            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100_000:
                print(f"    SKIP (exists): {fname}")
                total_skipped += 1
                continue

            print(f"    Downloading : {fname}")
            ok = download_file(bucket, key, dest_path)
            if ok:
                size_mb = os.path.getsize(dest_path) / 1e6
                print(f"      -> {size_mb:.1f} MB")
                total_downloaded += 1
            else:
                total_errors += 1

print(f"\n{'='*60}")
print(f"DOWNLOAD COMPLETE")
print(f"  Downloaded : {total_downloaded}")
print(f"  Skipped    : {total_skipped}  (already existed)")
print(f"  Errors     : {total_errors}")
print(f"{'='*60}")
print("\nNext step:  python coordinate.py")
