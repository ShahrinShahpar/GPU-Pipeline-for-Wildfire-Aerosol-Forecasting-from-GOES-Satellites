# save_of_reference.R
# ---------------------------------------------------------------------------
# Runs R's SpatialVx::OF(final, xhat=initial, W=15) for 20 G16 image pairs
# and saves the resulting speed and angle (in degrees) as 60x60 CSVs.
#
# Uses GPU KNN output CSVs as input (same float64 values as the GPU pipeline).
#
# Mirrors application.R lines 69-78:
#   for i in 1:20: of.fit <- OF(final=G16[i+1], xhat=G16[i], W=15)
#                  speed  <- matrix(of.fit$err.mag.lin, Nr, Nr)
#                  angle  <- matrix(of.fit$err.ang.lin, Nr, Nr)  ← DEGREES
#
# Run from gpuimplementation/ root:
#   Rscript r_reference/save_of_reference.R
#
# Input:  output/G16_knn/img_001.csv ... img_021.csv
# Output: validation/of/ref_G16/speed_001.csv ... speed_020.csv
#         validation/of/ref_G16/angle_001.csv ... angle_020.csv
#         Angles are saved in DEGREES (0..360), before /180*pi conversion.
#         (GPU optical_flow_gpu outputs degrees; Python validator compares as-is.)
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(SpatialVx)
})

KNN_DIR <- file.path("output", "G16_knn")
OUT_DIR <- file.path("validation", "of", "ref_G16")
Nr      <- 60
N_PAIRS <- 20   # pairs (1,2), (2,3), ..., (20,21)

if (!dir.exists(KNN_DIR)) {
  stop(paste("KNN output dir not found:", KNN_DIR,
             "\nRun ./aod_pipeline first."))
}
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("Reading GPU KNN CSVs from: %s\n", KNN_DIR))
cat(sprintf("Writing OF reference to:  %s\n", OUT_DIR))
cat(sprintf("Running OF (W=15) for %d pairs...\n\n", N_PAIRS))

# Load all required images (steps 1..21)
mats <- vector("list", N_PAIRS + 1)
for (i in seq_len(N_PAIRS + 1)) {
  path <- file.path(KNN_DIR, sprintf("img_%03d.csv", i))
  if (!file.exists(path))
    stop(sprintf("Missing KNN CSV: %s", path))
  df <- read.csv(path, header = TRUE, check.names = FALSE)
  mats[[i]] <- as.matrix(df)
  storage.mode(mats[[i]]) <- "double"
}
cat(sprintf("  Loaded %d images (%dx%d each).\n\n", N_PAIRS + 1, Nr, Nr))

t_start <- proc.time()["elapsed"]

for (i in seq_len(N_PAIRS)) {
  initial <- mats[[i]]
  final   <- mats[[i + 1]]

  cat(sprintf("  Pair (%02d,%02d) ... ", i, i + 1))
  of.fit <- OF(final, xhat = initial, W = 15, verbose = FALSE)

  # Speed: pixels/step (same as GPU output from optical_flow_gpu)
  speed <- matrix(of.fit$err.mag.lin, Nr, Nr)

  # Angle: DEGREES 0..360 (same as GPU output — do NOT divide by 180*pi here)
  angle <- matrix(of.fit$err.ang.lin, Nr, Nr)

  sp_path <- file.path(OUT_DIR, sprintf("speed_%03d.csv", i))
  an_path <- file.path(OUT_DIR, sprintf("angle_%03d.csv", i))
  write.csv(speed, file = sp_path, row.names = FALSE)
  write.csv(angle, file = an_path, row.names = FALSE)

  cat(sprintf("done -> %s\n", basename(sp_path)))
}

elapsed <- proc.time()["elapsed"] - t_start
cat(sprintf("\nDone. %d pairs written to %s\n", N_PAIRS, OUT_DIR))
cat(sprintf("Total time: %.1f s\n", elapsed))
cat("\nNext:\n  python validation/of/08_check_of.py\n")
