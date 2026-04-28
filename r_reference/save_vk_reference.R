# save_vk_reference.R
# ---------------------------------------------------------------------------
# Replicates application.R lines 78-80 and 111-126 exactly.
# Uses GPU smooth outputs as input so GPU and R start from identical data.
#
# Run from gpuimplementation/ root:
#   Rscript r_reference/save_vk_reference.R
#
# Input:  output/G16_smooth/speed_smooth.csv   (GPU output, Nr×Nr matrix)
#         output/G16_smooth/angle_smooth.csv   (GPU output, Nr×Nr matrix)
# Output: validation/k/ref_G16/v_a.csv         (long,lat,v_x,v_y)
#         validation/k/ref_G16/K_ifm.csv        (long,lat,K,dK_dx,dK_dy)
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(fields)
  library(matrixStats)   # rowDiffs, colDiffs
})

SMOOTH_DIR <- file.path("output", "G16_smooth")
OUT_DIR    <- file.path("validation", "k", "ref_G16")
Nr         <- 60

if (!dir.exists(SMOOTH_DIR))
  stop(paste("GPU smooth dir not found:", SMOOTH_DIR,
             "\nRun ./aod_pipeline first."))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ── load GPU smooth outputs ──────────────────────────────────────────────────
read_matrix <- function(path) {
  m <- as.matrix(read.csv(path, check.names = FALSE))
  storage.mode(m) <- "double"
  m   # Nr × Nr, row = lon, col = lat
}

speed_smooth <- read_matrix(file.path(SMOOTH_DIR, "speed_smooth.csv"))
angle_smooth <- read_matrix(file.path(SMOOTH_DIR, "angle_smooth.csv"))

cat(sprintf("Loaded speed_smooth: range [%.6g, %.6g]\n",
            min(speed_smooth, na.rm=TRUE), max(speed_smooth, na.rm=TRUE)))
cat(sprintf("Loaded angle_smooth: range [%.6g, %.6g]\n",
            min(angle_smooth, na.rm=TRUE), max(angle_smooth, na.rm=TRUE)))

# ── build grid coordinates (same as application.R) ──────────────────────────
lon_vec <- -123.98 + (0:(Nr-1)) * 0.04
lat_vec <-   35.02 + (0:(Nr-1)) * 0.04
long    <- matrix(rep(lon_vec, Nr),       Nr, Nr)   # row = lon, col = lat
lat     <- matrix(rep(lat_vec, each = Nr), Nr, Nr)

# ── velocity (application.R lines 78-80) ─────────────────────────────────────
# No unit conversion: v = speed_smooth * cos/sin(angle_smooth)
v.a.x <- c(speed_smooth) * cos(c(angle_smooth))
v.a.y <- c(speed_smooth) * sin(c(angle_smooth))

v.a.df <- data.frame(long = c(long), lat = c(lat), v_x = v.a.x, v_y = v.a.y)
write.csv(v.a.df, file.path(OUT_DIR, "v_a.csv"), row.names = FALSE)
cat(sprintf("Saved v_a.csv  v_x [%.6g, %.6g]  v_y [%.6g, %.6g]\n",
            min(v.a.x, na.rm=TRUE), max(v.a.x, na.rm=TRUE),
            min(v.a.y, na.rm=TRUE), max(v.a.y, na.rm=TRUE)))

# ── K diffusivity (application.R lines 111-126) ──────────────────────────────
# Normalised grid: delta = 1/Nr; dividing by delta = multiplying by Nr
v.a.x.m <- matrix(v.a.x, Nr, Nr)
v.a.y.m <- matrix(v.a.y, Nr, Nr)

delta <- 1.0 / Nr   # = 1/60 normalised grid spacing

col.dif.x <- rowDiffs(v.a.x.m) / delta   # rowDiffs(vx) * Nr
row.dif.x <- colDiffs(v.a.x.m) / delta   # colDiffs(vx) * Nr
col.dif.y <- rowDiffs(v.a.y.m) / delta
row.dif.y <- colDiffs(v.a.y.m) / delta

p1 <- cbind(col.dif.x, rep(NA, Nr))   # lat-dir diff of vx, pad last col
p2 <- rbind(row.dif.y, rep(NA, Nr))   # lon-dir diff of vy, pad last row
p3 <- rbind(row.dif.x, rep(NA, Nr))   # lon-dir diff of vx, pad last row
p4 <- cbind(col.dif.y, rep(NA, Nr))   # lat-dir diff of vy, pad last col

K <- 0.28 * delta * delta * sqrt((p1 - p2)^2 + (p3 + p4)^2)
K.smooth <- image.smooth(K)$z   # fields::image.smooth, matches GPU cuFFT

K.mat   <- matrix(K.smooth, Nr, Nr)

cat(sprintf("K range [%.6g, %.6g]  K.smooth range [%.6g, %.6g]\n",
            min(K, na.rm=TRUE), max(K, na.rm=TRUE),
            min(K.smooth, na.rm=TRUE), max(K.smooth, na.rm=TRUE)))

# ── dK gradients (application.R lines 122-125) ────────────────────────────────
# Repeat last valid difference at boundary (cbind/rbind with col/row 59)
col.dif <- rowDiffs(K.mat) / delta   # Nr × (Nr-1)
row.dif <- colDiffs(K.mat) / delta   # (Nr-1) × Nr

D.K.x <- cbind(col.dif, col.dif[, Nr-1])    # repeat last col
D.K.y <- rbind(row.dif, row.dif[Nr-1, ])    # repeat last row

K.ifm.df <- data.frame(
  long   = c(long),
  lat    = c(lat),
  K      = c(K.mat),
  dK_dx  = c(D.K.x),
  dK_dy  = c(D.K.y)
)
write.csv(K.ifm.df, file.path(OUT_DIR, "K_ifm.csv"), row.names = FALSE)
cat(sprintf("Saved K_ifm.csv  K [%.6g, %.6g]\n",
            min(K.ifm.df$K, na.rm=TRUE), max(K.ifm.df$K, na.rm=TRUE)))

cat("\nDone. Outputs in:", OUT_DIR, "\n")
