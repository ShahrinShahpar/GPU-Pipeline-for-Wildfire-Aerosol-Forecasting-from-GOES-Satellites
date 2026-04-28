#!/bin/bash
# =============================================================================
# run_pipeline.sh — GPU AOD pipeline (steps 1-6, Layer 1)
#
# Runs all steps in sequence. Each step is gated — it will not proceed if the
# previous step did not produce expected output.
#
# Usage:
#   bash run_pipeline.sh              # full run
#   bash run_pipeline.sh --skip-download  # skip step 1 if data already exists
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

SKIP_DOWNLOAD=0
for arg in "$@"; do
    [[ "$arg" == "--skip-download" ]] && SKIP_DOWNLOAD=1
done

log() { echo -e "\n$(date '+%H:%M:%S')  $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# STEP 1 — Download raw NC files
# ---------------------------------------------------------------------------
if [[ $SKIP_DOWNLOAD -eq 1 ]]; then
    log "STEP 1: Skipping download (--skip-download)"
else
    log "STEP 1: Downloading GOES-16 and GOES-17 NC files ..."
    python 01_download_data.py
fi

# Verify
G16_COUNT=$(find data/AOD/GOES-16/275 -name "*.nc" 2>/dev/null | wc -l)
G17_COUNT=$(find data/AOD/GOES-17/275 -name "*.nc" 2>/dev/null | wc -l)
log "  GOES-16: $G16_COUNT NC files   GOES-17: $G17_COUNT NC files"
[[ $G16_COUNT -ge 60 ]] || die "Expected >= 60 GOES-16 NC files, found $G16_COUNT"
[[ $G17_COUNT -ge 60 ]] || die "Expected >= 60 GOES-17 NC files, found $G17_COUNT"
log "STEP 1: PASS"

# ---------------------------------------------------------------------------
# STEP 2 — Generate coordinate lookup tables
# ---------------------------------------------------------------------------
log "STEP 2: Generating coordinate lookup tables ..."
python 02_coordinate.py

for f in data/AOD/g16_lats.txt data/AOD/g16_lons.txt \
          data/AOD/g17_lats.txt data/AOD/g17_lons.txt; do
    [[ -f "$f" ]] || die "Missing coordinate file: $f"
done
log "STEP 2: PASS  (4 coordinate txt files written)"

# ---------------------------------------------------------------------------
# STEP 3 — Preprocess NC -> binary
# ---------------------------------------------------------------------------
log "STEP 3: Preprocessing NC files -> binary ..."
python 03_preprocess.py

G16_BIN=$(find bin_images/G16 -name "*.bin" 2>/dev/null | wc -l)
G17_BIN=$(find bin_images/G17 -name "*.bin" 2>/dev/null | wc -l)
[[ -f manifest_G16.txt ]] || die "manifest_G16.txt not found"
[[ -f manifest_G17.txt ]] || die "manifest_G17.txt not found"
log "  bin_images/G16: $G16_BIN files   bin_images/G17: $G17_BIN files"
log "STEP 3: PASS"

# ---------------------------------------------------------------------------
# STEP 4 — Compile and run GPU regrid
# ---------------------------------------------------------------------------
log "STEP 4: Compiling regrid_gpu ..."
make clean && make
[[ -x ./regrid_gpu ]] || die "regrid_gpu binary not found after make"

log "  Running GPU regrid for G16 ..."
./regrid_gpu manifest_G16.txt output/G16

log "  Running GPU regrid for G17 ..."
./regrid_gpu manifest_G17.txt output/G17

G16_CSV=$(find output/G16 -name "*.csv" 2>/dev/null | wc -l)
G17_CSV=$(find output/G17 -name "*.csv" 2>/dev/null | wc -l)
log "  output/G16: $G16_CSV CSVs   output/G17: $G17_CSV CSVs"
[[ $G16_CSV -ge 60 ]] || die "Expected >= 60 G16 CSVs, got $G16_CSV"
[[ $G17_CSV -ge 60 ]] || die "Expected >= 60 G17 CSVs, got $G17_CSV"

# Quick sanity: first CSV should have 3601 lines (header + 3600 rows)
LINES=$(wc -l < output/G16/img_001.csv)
[[ $LINES -eq 3601 ]] || echo "  WARNING: img_001.csv has $LINES lines (expected 3601)"
log "STEP 4: PASS"

# ---------------------------------------------------------------------------
# STEP 5 — Merge per-image CSVs into combined files
# ---------------------------------------------------------------------------
log "STEP 5: Merging per-image CSVs ..."
python 05_merge.py

for f in output/G16_processed.csv output/G17_processed.csv; do
    [[ -f "$f" ]] || die "Missing: $f"
    LINES=$(wc -l < "$f")
    [[ $LINES -eq 216001 ]] || echo "  WARNING: $f has $LINES lines (expected 216001)"
done
log "STEP 5: PASS"

# ---------------------------------------------------------------------------
# STEP 6 — Compare against freshstaart reference
# ---------------------------------------------------------------------------
log "STEP 6: Comparing against freshstaart reference ..."
python 06_compare.py

log "STEP 6: Done — check correlation above. Expect >= 0.85 (archive drift OK)."

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline complete. Layer 1 (regrid) is done."
echo "  Next: add KNN imputation from gibbsgpu (Layer 2)."
echo "============================================================"
