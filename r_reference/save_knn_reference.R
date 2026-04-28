# save_knn_reference.R
# ---------------------------------------------------------------------------
# Runs R's knn.impute (bnstruct) on the GPU regrid output and saves
# each time step's imputed 60x60 matrix as a CSV for comparison against the
# GPU KNN output.
#
# Reads GPU regrid CSVs (output/G16/img_NNN.csv) so R's KNN uses EXACTLY
# the same float64 values as the GPU KNN kernel.  This eliminates tiny ULP
# differences that arise when R's regrid and GPU regrid compute multi-
# observation means in different FP summation orders.
#
# Input:  output/G16/img_001.csv ... img_060.csv
#         Columns: long, lat, AOD
#         3600 rows per file, ordered g = lat_i*60 + lon_i  (lon fastest)
#         Written by regrid_standalone with %.17g precision.
#
# Run from gpuimplementation/ root:
#   Rscript r_reference/save_knn_reference.R
#
# Output:
#   validation/knn/ref_G16/img_001.csv ... img_060.csv
#   Each file: 60x60 matrix (no row names), columns V1..V60
#   Row r (0-indexed) = all latitudes at longitude r  (R's matrix() column-major fill)
#
# The GPU KNN standalone verifier and 07_check_knn.py expect exactly this format.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(bnstruct)
})

# ---------------------------------------------------------------------------
# Paths — run from gpuimplementation/ root
# ---------------------------------------------------------------------------
REGRID_DIR <- file.path("output", "G16")
OUT_DIR    <- file.path("validation", "knn", "ref_G16")
N_STEPS    <- 60
Nr         <- 60

if (!dir.exists(REGRID_DIR)) {
  stop(paste("Cannot find GPU regrid output dir:", REGRID_DIR,
             "\nRun ./regrid_standalone first (with %.17g AOD precision)."))
}

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("Reading GPU regrid CSVs from: %s\n", REGRID_DIR))
cat(sprintf("Writing KNN reference to:     %s\n", OUT_DIR))
cat(sprintf("Running knn.impute (k=10) on %d time steps...\n\n", N_STEPS))

t_start <- proc.time()["elapsed"]

for (i in seq_len(N_STEPS)) {
  in_path <- file.path(REGRID_DIR, sprintf("img_%03d.csv", i))

  if (!file.exists(in_path)) {
    warning(sprintf("  [%02d] SKIP — file not found: %s", i, in_path))
    next
  }

  if (i %% 10 == 0 || i == 1)
    cat(sprintf("  [%02d/%02d] %s\n", i, N_STEPS, in_path))

  # Read GPU regrid CSV: columns long, lat, AOD; 3600 rows in g=lat*60+lon order
  df <- read.csv(in_path, header = TRUE, na.strings = c("NA", "NaN", "nan"))

  if (nrow(df) != Nr * Nr)
    warning(sprintf("  Step %d: expected %d rows, got %d", i, Nr * Nr, nrow(df)))

  # Build 60x60 matrix identical to what main_pipeline.cu feeds to GPU KNN:
  #   matrix(AOD_vector, Nr, Nr) — column-major fill
  #   AOD_vector is in g = lat_i*Nr + lon_i order (lon fastest).
  #   Column-major fill → mat[lon_i+1, lat_i+1] = AOD[lat_i*Nr + lon_i].
  #   Row r = all latitudes at longitude r-1.
  tempt <- matrix(df$AOD, Nr, Nr)

  # R KNN imputation (exact paper parameters)
  imputed <- knn.impute(
    tempt,
    k         = 10,
    cat.var   = 1:ncol(tempt),
    to.impute = 1:nrow(tempt),
    using     = 1:nrow(tempt)
  )

  # Save: no row names, column names V1..V60 written by write.csv
  out_path <- file.path(OUT_DIR, sprintf("img_%03d.csv", i))
  write.csv(imputed, file = out_path, row.names = FALSE)
}

elapsed <- proc.time()["elapsed"] - t_start
cat(sprintf("\nDone. %d reference files written to %s\n", N_STEPS, OUT_DIR))
cat(sprintf("Total time: %.1f seconds\n", elapsed))
cat("\nVerify one file:\n")
cat(sprintf("  head -2 %s/img_001.csv\n", OUT_DIR))
cat("\nNext: compare GPU vs R:\n")
cat("  python validation/knn/07_check_knn.py\n")
