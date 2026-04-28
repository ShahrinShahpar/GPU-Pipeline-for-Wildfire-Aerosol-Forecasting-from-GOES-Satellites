# save_smooth_reference.R
# ---------------------------------------------------------------------------
# Replicates R's velocity(dat.velocity, 1, 19) function (util.R line 612):
#   1. Load OF speed/angle CSVs for pairs 1..19 (from validation/of/ref_G16/)
#   2. Convert angle to radians (matches dat.velocity storage)
#   3. Average speed and angle across 19 pairs
#   4. image.smooth(speed.avg)$z / N_r  → speed_smooth
#   5. image.smooth(angle.avg)$z         → angle_smooth
#
# Run from gpuimplementation/ root:
#   Rscript r_reference/save_smooth_reference.R
#
# Input:  validation/of/ref_G16/speed_001..019.csv
#         validation/of/ref_G16/angle_001..019.csv
# Output: validation/smooth/ref_G16/speed_smooth.csv
#         validation/smooth/ref_G16/angle_smooth.csv
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(fields)
})

OF_DIR  <- file.path("validation", "of",     "ref_G16")
OUT_DIR <- file.path("validation", "smooth",  "ref_G16")
Nr      <- 60
N_AVG   <- 19   # pairs 1..19 (R: velocity(dat.velocity, 1, 19))

if (!dir.exists(OF_DIR)) {
  stop(paste("OF reference dir not found:", OF_DIR,
             "\nRun: Rscript r_reference/save_of_reference.R"))
}
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("Reading OF reference from: %s (pairs 1..%d)\n", OF_DIR, N_AVG))
cat(sprintf("Writing smooth reference to: %s\n\n", OUT_DIR))

# Accumulate speed and angle across N_AVG pairs
speed_sum <- matrix(0.0, Nr, Nr)
angle_sum <- matrix(0.0, Nr, Nr)  # in radians
has_nan   <- matrix(FALSE, Nr, Nr)

for (i in seq_len(N_AVG)) {
  sp_path <- file.path(OF_DIR, sprintf("speed_%03d.csv", i))
  an_path <- file.path(OF_DIR, sprintf("angle_%03d.csv", i))

  sp_df <- read.csv(sp_path, header = TRUE, check.names = FALSE)
  an_df <- read.csv(an_path, header = TRUE, check.names = FALSE)

  sp_mat <- as.matrix(sp_df); storage.mode(sp_mat) <- "double"
  an_mat <- as.matrix(an_df); storage.mode(an_mat) <- "double"

  # Mark cells where either speed or angle is NA/NaN (boundary cells)
  has_nan <- has_nan | is.na(sp_mat) | is.na(an_mat)

  # Replace NA with 0 for summation (will be overwritten with NA later)
  sp_mat[is.na(sp_mat)] <- 0
  an_mat[is.na(an_mat)] <- 0

  speed_sum <- speed_sum + sp_mat
  angle_sum <- angle_sum + an_mat / 180 * pi   # degrees → radians
}

speed_avg <- speed_sum / N_AVG
angle_avg <- angle_sum / N_AVG

# Restore NA for cells with any NaN pair
speed_avg[has_nan] <- NA
angle_avg[has_nan] <- NA

cat("Averaged 19 pairs. Running image.smooth...\n")

# Smooth — default parameters (aRange=1, matches R's fields::image.smooth defaults)
speed_smooth_mat <- image.smooth(speed_avg)$z / Nr   # divide by Nr=60
angle_smooth_mat <- image.smooth(angle_avg)$z

cat(sprintf("  speed_smooth: range [%.4f, %.4f]\n",
            min(speed_smooth_mat, na.rm=TRUE), max(speed_smooth_mat, na.rm=TRUE)))
cat(sprintf("  angle_smooth: range [%.4f, %.4f]\n",
            min(angle_smooth_mat, na.rm=TRUE), max(angle_smooth_mat, na.rm=TRUE)))

sp_out <- file.path(OUT_DIR, "speed_smooth.csv")
an_out <- file.path(OUT_DIR, "angle_smooth.csv")
write.csv(speed_smooth_mat, file = sp_out, row.names = FALSE)
write.csv(angle_smooth_mat, file = an_out, row.names = FALSE)

cat(sprintf("\nWritten:\n  %s\n  %s\n", sp_out, an_out))
cat("\nNext:\n  python validation/smooth/09_check_smooth.py\n")
