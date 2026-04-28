// knn_impute.cu
// GPU implementation of bnstruct::knn.impute — exact categorical HEOM distance.
//
// R call in application.R:
//   knn.impute(tempt, k=10, cat.var=1:ncol(tempt),
//              to.impute=1:nrow(tempt), using=1:nrow(tempt))
//
// Build as standalone verifier:
//   nvcc -arch=sm_70 -std=c++14 -O3 -I. -DKNN_VERIFY_MAIN \
//        knn_impute.cu csv_io.cu -o knn_impute_verify
//
// Usage:
//   ./knn_impute_verify input.csv ref.csv output.csv

#include "knn_impute.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Kernel: one CUDA thread per row (row = R's observation = one spatial row).
//
// Distance metric: categorical HEOM
//   cat_dist(a, b) = 0  if a == b (exact, both non-NaN)
//                  = 1  otherwise (differ, or either NaN)
//   d(r, s) = sqrt( sum_c cat_dist(r[c], s[c]) )
//
// For each missing column c in row r:
//   Walk sorted neighbor list, collect k non-NaN values, impute with mode.
//   For continuous floats (all values typically distinct), mode = nearest
//   neighbor's value (matches R's stat.mode.ord behavior).
//
// Shared memory: snapshot of the original data so imputed writes to d_aod
//   by one thread don't affect distance computations of other threads.
// ---------------------------------------------------------------------------
static __global__ void knn_impute_kernel(double* __restrict__ d_aod,
                                         int Nr, int k)
{
    extern __shared__ double s_aod[];   // Nr * Nr doubles (read-only snapshot)

    int tid   = threadIdx.x;
    int total = Nr * Nr;

    // All 64 threads cooperatively copy global → shared
    for (int i = tid; i < total; i += blockDim.x)
        s_aod[i] = d_aod[i];
    __syncthreads();

    if (tid >= Nr) return;
    int r = tid;

    // Skip rows with no missing values
    bool has_nan = false;
    for (int c = 0; c < Nr; c++) {
        if (isnan(s_aod[r * Nr + c])) { has_nan = true; break; }
    }
    if (!has_nan) return;

    // ---- Categorical HEOM distances -----------------------------------
    // d^2(r, s) = count of columns where cat_dist == 1
    // cat_dist(a,b) = 1 if isnan(a) || isnan(b) || a != b, else 0
    // We store integer squared distances (= counts), range [0, Nr].
    int idists[64];
    for (int s = 0; s < Nr; s++) {
        int cnt = 0;
        for (int c = 0; c < Nr; c++) {
            double vr = s_aod[r * Nr + c];
            double vs = s_aod[s * Nr + c];
            if (isnan(vr) || isnan(vs) || vr != vs)
                ++cnt;
        }
        idists[s] = cnt;
    }
    // Self-distance set to maximum so it sorts last
    idists[r] = Nr + 1;

    // ---- Sort row indices by ascending HEOM distance (insertion sort) ---
    // Nr = 60, so 60-element insertion sort is fast in registers.
    int sorted[64];
    for (int i = 0; i < Nr; i++) sorted[i] = i;
    for (int i = 1; i < Nr; i++) {
        int key   = sorted[i];
        int key_d = idists[key];
        int j = i - 1;
        while (j >= 0 && idists[sorted[j]] > key_d) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }

    // ---- Per-column imputation -----------------------------------------
    // For each missing column c:
    //   Walk sorted rows, collect values from k rows that have non-NaN at c.
    //   Compute mode (stat.mode.ord): for continuous floats = nearest value.
    for (int c = 0; c < Nr; c++) {
        if (!isnan(s_aod[r * Nr + c])) continue;

        // Collect up to k neighbor values at column c (in distance order)
        double vals[16];
        int    cnt = 0;
        for (int si = 0; si < Nr && cnt < k; si++) {
            int    s = sorted[si];
            double v = s_aod[s * Nr + c];
            if (!isnan(v))
                vals[cnt++] = v;
        }
        if (cnt == 0) continue;   // no usable neighbor — leave NaN

        // stat.mode.ord: return first unique value with highest frequency.
        // unique(x) preserves order of first appearance.
        double best_val  = vals[0];
        int    best_freq = 0;
        for (int a = 0; a < cnt; a++) {
            // Only consider first occurrence of each distinct value
            bool first_occ = true;
            for (int b = 0; b < a; b++) {
                if (vals[b] == vals[a]) { first_occ = false; break; }
            }
            if (!first_occ) continue;
            // Count frequency of vals[a]
            int freq = 0;
            for (int b = 0; b < cnt; b++)
                if (vals[b] == vals[a]) freq++;
            if (freq > best_freq) {
                best_freq = freq;
                best_val  = vals[a];
            }
        }
        d_aod[r * Nr + c] = best_val;
    }
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
void knn_impute_gpu(double* d_aod, int Nr, int k)
{
    if (Nr > 64)
        throw std::runtime_error("knn_impute_gpu: Nr > 64 not supported");
    if (k > 16)
        throw std::runtime_error("knn_impute_gpu: k > 16 not supported");

    int    threads = 64;
    size_t smem    = (size_t)Nr * Nr * sizeof(double);

    knn_impute_kernel<<<1, threads, smem>>>(d_aod, Nr, k);

    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("knn_impute_kernel failed: ") + cudaGetErrorString(err));
}

// ===========================================================================
// Standalone verifier — compiled only when -DKNN_VERIFY_MAIN is set
// ===========================================================================
#ifdef KNN_VERIFY_MAIN

#include "csv_io.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

static int count_nan(const std::vector<double>& v) {
    int n = 0;
    for (double x : v) if (std::isnan(x)) ++n;
    return n;
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s input.csv ref.csv output.csv\n\n"
            "  input.csv  — raw 60x60 matrix with NaN/NA (from R write.csv)\n"
            "  ref.csv    — R knn.impute reference, 60x60 (from R write.csv)\n"
            "  output.csv — GPU result written here (no header)\n",
            argv[0]);
        return 1;
    }

    const char* input_path  = argv[1];
    const char* ref_path    = argv[2];
    const char* output_path = argv[3];

    const int Nr = 60;
    const int k  = 10;

    // ---- Read raw input ---------------------------------------------------
    int nrows = 0, ncols = 0;
    std::vector<double> h_aod;
    try {
        h_aod = read_csv_flat(input_path, nrows, ncols, /*has_header=*/true);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading input '%s': %s\n", input_path, e.what());
        return 1;
    }
    if (nrows != Nr || ncols != Nr) {
        fprintf(stderr, "ERROR: expected %dx%d, input is %dx%d\n",
                Nr, Nr, nrows, ncols);
        return 1;
    }
    int nan_before = count_nan(h_aod);
    printf("Input  : %s  (%dx%d, %d NaN cells)\n",
           input_path, nrows, ncols, nan_before);

    // ---- GPU imputation ---------------------------------------------------
    double* d_aod  = nullptr;
    size_t  nbytes = (size_t)Nr * Nr * sizeof(double);

    cudaError_t cerr = cudaMalloc(&d_aod, nbytes);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cerr));
        return 1;
    }
    cudaMemcpy(d_aod, h_aod.data(), nbytes, cudaMemcpyHostToDevice);

    try {
        knn_impute_gpu(d_aod, Nr, k);
    } catch (const std::exception& e) {
        fprintf(stderr, "GPU error: %s\n", e.what());
        cudaFree(d_aod);
        return 1;
    }

    std::vector<double> h_result(Nr * Nr);
    cudaMemcpy(h_result.data(), d_aod, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_aod);

    int nan_after = count_nan(h_result);
    printf("GPU out: %d NaN cells remain after imputation\n", nan_after);

    // ---- Write GPU output -------------------------------------------------
    try {
        write_csv_flat(output_path, h_result.data(), Nr, Nr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR writing output '%s': %s\n", output_path, e.what());
        return 1;
    }
    printf("Written: %s\n", output_path);

    // ---- Read R reference -------------------------------------------------
    int ref_rows = 0, ref_cols = 0;
    std::vector<double> h_ref;
    try {
        h_ref = read_csv_flat(ref_path, ref_rows, ref_cols, /*has_header=*/true);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading reference '%s': %s\n", ref_path, e.what());
        return 1;
    }
    if (ref_rows != Nr || ref_cols != Nr) {
        fprintf(stderr, "ERROR: reference expected %dx%d, got %dx%d\n",
                Nr, Nr, ref_rows, ref_cols);
        return 1;
    }

    // ---- Field-by-field comparison ----------------------------------------
    int    total_cells    = Nr * Nr;
    int    nan_mismatch   = 0;
    int    cells_compared = 0;
    double max_diff       = 0.0;
    double sum_diff       = 0.0;
    int    gt_1e8         = 0;

    struct BadCell { int idx; double diff; };
    std::vector<BadCell> bad_cells;
    bad_cells.reserve(total_cells);

    for (int i = 0; i < total_cells; i++) {
        bool gpu_nan = std::isnan(h_result[i]);
        bool ref_nan = std::isnan(h_ref[i]);

        if (gpu_nan || ref_nan) {
            if (gpu_nan != ref_nan) ++nan_mismatch;
            continue;
        }
        double diff = std::fabs(h_result[i] - h_ref[i]);
        max_diff  = std::max(max_diff, diff);
        sum_diff += diff;
        ++cells_compared;
        if (diff > 1e-8) ++gt_1e8;
        bad_cells.push_back({i, diff});
    }
    double mean_diff = (cells_compared > 0) ? sum_diff / cells_compared : 0.0;

    std::sort(bad_cells.begin(), bad_cells.end(),
              [](const BadCell& a, const BadCell& b){ return a.diff > b.diff; });

    printf("\n=== KNN Imputation Verification ===\n");
    printf("Total cells        : %d\n",   total_cells);
    printf("Cells compared     : %d\n",   cells_compared);
    printf("NaN mismatches     : %d\n",   nan_mismatch);
    printf("Max  |GPU - R|     : %.3e\n", max_diff);
    printf("Mean |GPU - R|     : %.3e\n", mean_diff);
    printf("Cells > 1e-8       : %d\n",   gt_1e8);

    int show = (int)std::min((size_t)5, bad_cells.size());
    if (show > 0 && bad_cells[0].diff > 1e-10) {
        printf("\nTop-%d worst-diff cells (row col GPU_value R_value diff):\n", show);
        for (int i = 0; i < show; i++) {
            int idx = bad_cells[i].idx;
            int row = idx / Nr;
            int col = idx % Nr;
            printf("  [%2d,%2d]  GPU=%.10g  R=%.10g  |diff|=%.3e\n",
                   row, col, h_result[idx], h_ref[idx], bad_cells[i].diff);
        }
    }

    bool pass = (nan_mismatch == 0) && (max_diff < 1e-6);
    printf("\nResult : %s\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        if (nan_mismatch > 0)
            printf("  Reason: %d NaN/NA mismatches between GPU and R\n",
                   nan_mismatch);
        if (max_diff >= 1e-6)
            printf("  Reason: max |diff| = %.3e >= 1e-6\n", max_diff);
    }

    return pass ? 0 : 1;
}

#endif // KNN_VERIFY_MAIN
