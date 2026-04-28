// obs_assembly.cu — GPU observation assembly (M_id + row-selection of F).
// Replicates util.R::OBS_ccl2() for one time step.
//
// Build as standalone verifier:
//   nvcc -arch=sm_70 -std=c++14 -O3 -I. -DOBS_VERIFY_MAIN \
//        obs_assembly.cu csv_io.cu -o obs_assembly_verify
//
// Usage:
//   ./obs_assembly_verify F.csv y.csv ref_Ft.csv ref_yc.csv ref_id.csv

#include "obs_assembly.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Device kernel: gather rows of F
//   For k in 0..n_obs-1, j in 0..Ngrid-1:
//     d_Ft[k * Ngrid + j] = d_F[d_id[k] * Ngrid + j]
// ---------------------------------------------------------------------------
static __global__ void gather_rows_kernel(double* d_Ft, const double* d_F,
                                           const int* d_id,
                                           int n_obs, int Ngrid)
{
    int k = blockIdx.x;         // observation index
    int j_base = threadIdx.x;   // column offset within a block
    if (k >= n_obs) return;
    int src_row = d_id[k];
    for (int j = j_base; j < Ngrid; j += blockDim.x)
        d_Ft[k * Ngrid + j] = d_F[src_row * Ngrid + j];
}

// ---------------------------------------------------------------------------
// Functor must live at namespace scope for CUDA 11 Thrust compatibility
// ---------------------------------------------------------------------------
struct not_nan_pred {
    __device__ bool operator()(double v) const { return !isnan(v); }
};

// ---------------------------------------------------------------------------
// obs_assembly_gpu
// ---------------------------------------------------------------------------
void obs_assembly_gpu(const double* d_F, const double* d_y,
                      int Nr2, int Ngrid,
                      double* d_Ft, double* d_yc, int* d_id,
                      int& n_obs)
{

    thrust::device_ptr<const double> y_ptr(d_y);
    thrust::device_ptr<double>       yc_ptr(d_yc);
    thrust::device_ptr<int>          id_ptr(d_id);

    // Extract non-NaN indices (0-based)
    auto id_end = thrust::copy_if(
        thrust::device,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(Nr2),
        y_ptr,          // stencil: apply predicate to y values
        id_ptr,
        not_nan_pred{}
    );
    n_obs = (int)(id_end - id_ptr);

    if (n_obs == 0) return;

    // Extract the non-NaN y values
    thrust::gather(thrust::device,
                   id_ptr, id_ptr + n_obs,
                   y_ptr, yc_ptr);

    // Extract corresponding rows of F
    const int T = 256;
    gather_rows_kernel<<<n_obs, T>>>(d_Ft, d_F, d_id, n_obs, Ngrid);
    cudaDeviceSynchronize();
}

// ===========================================================================
// Standalone verifier — compiled only when -DOBS_VERIFY_MAIN is set
// ===========================================================================
#ifdef OBS_VERIFY_MAIN

#include "csv_io.cuh"
#include <cstdio>
#include <algorithm>
#include <cstring>

// Read a single-column CSV (with header) as a vector
static std::vector<double> read_col_csv(const char* path) {
    int nrows = 0, ncols = 0;
    auto flat = read_csv_flat(path, nrows, ncols, true);
    std::vector<double> v(nrows);
    for (int i = 0; i < nrows; i++) v[i] = flat[i * ncols];
    return v;
}

int main(int argc, char* argv[])
{
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s F.csv y.csv ref_Ft.csv ref_yc.csv ref_id.csv\n\n"
            "  F.csv      — Nr²×Ngrid Fourier basis (no header)\n"
            "  y.csv      — Nr² AOD observations with NaN (1 column, with header)\n"
            "  ref_Ft.csv — R's Ft (n_obs×Ngrid, with header)\n"
            "  ref_yc.csv — R's yc (n_obs values, with header)\n"
            "  ref_id.csv — R's id (n_obs indices, 1-indexed, with header)\n",
            argv[0]);
        return 1;
    }

    // ---- Read inputs -------------------------------------------------------
    int F_rows = 0, F_cols = 0;
    std::vector<double> h_F, h_y;
    try {
        h_F = read_csv_flat(argv[1], F_rows, F_cols, false);  // no header
        h_y = read_col_csv(argv[2]);                           // 1-column with header
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading input: %s\n", e.what());
        return 1;
    }
    int Nr2   = F_rows;
    int Ngrid = F_cols;
    printf("F: %d × %d   |   y: %d values\n", Nr2, Ngrid, (int)h_y.size());

    // ---- Upload to device --------------------------------------------------
    double *d_F = nullptr, *d_y = nullptr;
    double *d_Ft = nullptr, *d_yc = nullptr;
    int    *d_id = nullptr;

    cudaMalloc(&d_F,  (size_t)Nr2 * Ngrid * sizeof(double));
    cudaMalloc(&d_y,  Nr2 * sizeof(double));
    cudaMalloc(&d_Ft, (size_t)Nr2 * Ngrid * sizeof(double));
    cudaMalloc(&d_yc, Nr2 * sizeof(double));
    cudaMalloc(&d_id, Nr2 * sizeof(int));

    cudaMemcpy(d_F, h_F.data(), (size_t)Nr2 * Ngrid * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), Nr2 * sizeof(double), cudaMemcpyHostToDevice);

    // ---- Run obs_assembly_gpu ---------------------------------------------
    int n_obs = 0;
    obs_assembly_gpu(d_F, d_y, Nr2, Ngrid, d_Ft, d_yc, d_id, n_obs);
    printf("n_obs (GPU) = %d\n", n_obs);

    // ---- Download results --------------------------------------------------
    std::vector<double> h_Ft(n_obs * Ngrid), h_yc(n_obs);
    std::vector<int>    h_id(n_obs);
    cudaMemcpy(h_Ft.data(), d_Ft, (size_t)n_obs * Ngrid * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_yc.data(), d_yc, n_obs * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_id.data(), d_id, n_obs * sizeof(int),    cudaMemcpyDeviceToHost);

    cudaFree(d_F); cudaFree(d_y); cudaFree(d_Ft); cudaFree(d_yc); cudaFree(d_id);

    // ---- Read R references -------------------------------------------------
    int ref_Ft_rows = 0, ref_Ft_cols = 0;
    std::vector<double> h_ref_Ft, h_ref_yc, h_ref_id_d;
    try {
        h_ref_Ft  = read_csv_flat(argv[3], ref_Ft_rows, ref_Ft_cols, true);
        h_ref_yc  = read_col_csv(argv[4]);
        h_ref_id_d = read_col_csv(argv[5]);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading reference: %s\n", e.what());
        return 1;
    }
    int n_obs_R = ref_Ft_rows;
    printf("n_obs (R)   = %d\n", n_obs_R);

    if (n_obs != n_obs_R) {
        printf("\nFAIL: n_obs mismatch: GPU=%d  R=%d\n", n_obs, n_obs_R);
        return 1;
    }

    // Convert R 1-indexed ids to 0-indexed
    std::vector<int> h_ref_id(n_obs_R);
    for (int i = 0; i < n_obs_R; i++)
        h_ref_id[i] = (int)h_ref_id_d[i] - 1;   // R is 1-indexed

    // ---- Compare ids -------------------------------------------------------
    int id_mismatch = 0;
    for (int i = 0; i < n_obs; i++)
        if (h_id[i] != h_ref_id[i]) ++id_mismatch;

    // ---- Compare yc --------------------------------------------------------
    double yc_max = 0.0, yc_sum = 0.0;
    for (int i = 0; i < n_obs; i++) {
        double d = std::fabs(h_yc[i] - h_ref_yc[i]);
        yc_max = std::max(yc_max, d);
        yc_sum += d;
    }

    // ---- Compare Ft --------------------------------------------------------
    double Ft_max = 0.0, Ft_sum = 0.0;
    int    Ft_gt_1e6 = 0;
    struct BadCell { int r, c; double d, g, ref; };
    std::vector<BadCell> bad;
    for (int i = 0; i < n_obs * Ngrid; i++) {
        double d = std::fabs(h_Ft[i] - h_ref_Ft[i]);
        Ft_max = std::max(Ft_max, d);
        Ft_sum += d;
        if (d > 1e-6) ++Ft_gt_1e6;
        if (d > 1e-8) bad.push_back({i / Ngrid, i % Ngrid, d, h_Ft[i], h_ref_Ft[i]});
    }
    std::sort(bad.begin(), bad.end(),
              [](const BadCell& a, const BadCell& b){ return a.d > b.d; });

    // ---- Report ------------------------------------------------------------
    printf("\n=== Obs Assembly Verification ===\n");
    printf("n_obs              : %d\n",   n_obs);
    printf("ID mismatches      : %d\n",   id_mismatch);
    printf("Max  |yc GPU - R|  : %.3e\n", yc_max);
    printf("Max  |Ft GPU - R|  : %.3e\n", Ft_max);
    printf("Mean |Ft GPU - R|  : %.3e\n", n_obs*Ngrid > 0 ? Ft_sum/(n_obs*Ngrid) : 0.0);
    printf("Ft cells > 1e-6    : %d\n",   Ft_gt_1e6);

    int show = (int)std::min((size_t)3, bad.size());
    if (show > 0) {
        printf("\nTop-%d worst Ft cells (row col GPU R diff):\n", show);
        for (int i = 0; i < show; i++)
            printf("  [%3d,%3d]  GPU=%.10g  R=%.10g  |diff|=%.3e\n",
                   bad[i].r, bad[i].c, bad[i].g, bad[i].ref, bad[i].d);
    }

    bool pass = (id_mismatch == 0) && (yc_max < 1e-10) && (Ft_max < 1e-10);
    printf("\nResult : %s\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        if (id_mismatch > 0) printf("  Reason: %d ID mismatches\n", id_mismatch);
        if (yc_max >= 1e-10) printf("  Reason: yc max |diff| = %.3e\n", yc_max);
        if (Ft_max >= 1e-10) printf("  Reason: Ft max |diff| = %.3e\n", Ft_max);
    }
    return pass ? 0 : 1;
}

#endif // OBS_VERIFY_MAIN
