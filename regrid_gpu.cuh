#pragma once

/*
 * regrid_gpu.cuh — callable GPU regrid module declaration.
 *
 * Wraps k_regrid kernel for use inside main_pipeline.cu.
 *
 * Output ordering: d_means[lat_i * N_LON + lon_i]  (lon varies fastest).
 * This matches R's expand.grid(long, lat) ordering.
 *
 * d_lats, d_lons, d_aods : device pointers, n_points doubles each
 * d_means                : device output, N_GRID = 3600 doubles (pre-allocated)
 *
 * Synchronizes before returning.
 */
void regrid_gpu(const double* d_lats,
                const double* d_lons,
                const double* d_aods,
                int           n_points,
                double*       d_means);
