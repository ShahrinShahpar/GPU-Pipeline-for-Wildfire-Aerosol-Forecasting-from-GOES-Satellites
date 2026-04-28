// image_smooth.cu
// GPU implementation of fields::image.smooth (double.exp kernel, FFT-based).
//
// Build as standalone verifier:
//   nvcc -arch=sm_70 -std=c++14 -O3 -I. -DSMOOTH_VERIFY_MAIN \
//        image_smooth.cu csv_io.cu -o image_smooth_verify -lcufft
//
// Usage:
//   ./image_smooth_verify input.csv ref.csv output.csv

#include "image_smooth.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Build double.exp kernel on M×N padded grid, centered at (cm, cn).
// double.exp(d2) = 0.5 * exp(-d2), d2 = ((m-cm)/aRange)^2 + ((n-cn)/aRange)^2
static __global__ void build_kernel_g(cufftDoubleComplex* out,
                                      int M, int N, int cm, int cn,
                                      double inv_aRange)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N, n = idx % N;
    double xi = (double)(m - cm) * inv_aRange;
    double yi = (double)(n - cn) * inv_aRange;
    double d2 = xi * xi + yi * yi;
    out[idx] = {0.5 * exp(-d2), 0.0};
}

// Build impulse at (cm, cn): 1 there, 0 elsewhere.
static __global__ void build_impulse_g(cufftDoubleComplex* out,
                                       int M, int N, int cm, int cn)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    out[idx] = {(idx == cm * N + cn) ? 1.0 : 0.0, 0.0};
}

// W[k] = Fkern[k] / Fimp[k] * inv_MN  (element-wise complex division + scale)
static __global__ void compute_W_g(cufftDoubleComplex* W,
                                   const cufftDoubleComplex* Fk,
                                   const cufftDoubleComplex* Fi,
                                   double inv_MN, int MN)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= MN) return;
    double c = Fi[k].x, d = Fi[k].y;
    double denom = c * c + d * d;
    double ar = Fk[k].x * inv_MN;
    double ai = Fk[k].y * inv_MN;
    W[k] = {(ar * c + ai * d) / denom,
             (ai * c - ar * d) / denom};
}

// Pad Nr×Nr input into M×N grid (top-left block, rest = 0).
// Simultaneously builds data_pad (NaN→0) and mask_pad (NaN→0, else→1).
static __global__ void pad_data_g(cufftDoubleComplex* data_pad,
                                   cufftDoubleComplex* mask_pad,
                                   const double* d_in,
                                   int Nr, int N, int MN)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MN) return;
    int r = idx / N, c = idx % N;
    if (r < Nr && c < Nr) {
        double v   = d_in[r * Nr + c];
        bool  isna = isnan(v);
        data_pad[idx] = {isna ? 0.0 : v,   0.0};
        mask_pad[idx] = {isna ? 0.0 : 1.0, 0.0};
    } else {
        data_pad[idx] = {0.0, 0.0};
        mask_pad[idx] = {0.0, 0.0};
    }
}

// Element-wise complex multiply: out[k] = A[k] × W[k]
static __global__ void cmul_g(cufftDoubleComplex* out,
                               const cufftDoubleComplex* A,
                               const cufftDoubleComplex* W,
                               int MN)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= MN) return;
    double ar = A[k].x, ai = A[k].y;
    double wr = W[k].x, wi = W[k].y;
    out[k] = {ar * wr - ai * wi, ar * wi + ai * wr};
}

// Extract top-left Nr×Nr block and apply Nadaraya-Watson division.
// Reads from M×N padded arrays (stride = N), writes to Nr×Nr (stride = Nr).
static __global__ void extract_g(double* d_out,
                                  const cufftDoubleComplex* num,
                                  const cufftDoubleComplex* den,
                                  int Nr, int N, double tol)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Nr || c >= Nr) return;
    double n_val = num[r * N + c].x;
    double d_val = den[r * N + c].x;
    d_out[r * Nr + c] = (d_val > tol) ? n_val / d_val : (double)NAN;
}

// ---------------------------------------------------------------------------
// Error-check helpers
// ---------------------------------------------------------------------------
static void chk_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
static void chk_cufft(cufftResult r, const char* where) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(where) +
                                 " (cufftResult=" + std::to_string((int)r) + ")");
}

// ---------------------------------------------------------------------------
// RAII wrapper for cuFFT plan
// ---------------------------------------------------------------------------
struct CufftPlan {
    cufftHandle h = 0;
    CufftPlan(int M, int N) {
        chk_cufft(cufftPlan2d(&h, M, N, CUFFT_Z2Z), "cufftPlan2d");
    }
    ~CufftPlan() { if (h) cufftDestroy(h); }
    void fwd(cufftDoubleComplex* io) {
        chk_cufft(cufftExecZ2Z(h, io, io, CUFFT_FORWARD), "fwd FFT");
    }
    void inv(cufftDoubleComplex* io) {
        chk_cufft(cufftExecZ2Z(h, io, io, CUFFT_INVERSE), "inv FFT");
    }
};

// RAII wrapper for device memory
struct DevBuf {
    void* ptr = nullptr;
    DevBuf(size_t bytes) { chk_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc"); }
    ~DevBuf() { if (ptr) cudaFree(ptr); }
    cufftDoubleComplex* c() { return (cufftDoubleComplex*)ptr; }
};

// ---------------------------------------------------------------------------
// Host wrapper — image_smooth_gpu
// ---------------------------------------------------------------------------
void image_smooth_gpu(const double* d_in, double* d_out, int Nr,
                      double aRange, double tol)
{
    // Padded dimensions matching fields::setup.image.smooth defaults:
    //   xwidth = Nr*dx = Nr,  M2 = round((Nr + Nr)/2) = Nr
    //   M = N = 2*Nr
    const int M   = 2 * Nr;
    const int N   = 2 * Nr;
    const int MN  = M * N;
    const int cm  = Nr - 1;   // 0-indexed kernel center row  (= M2-1 = 59)
    const int cn  = Nr - 1;   // 0-indexed kernel center col

    size_t bytes = (size_t)MN * sizeof(cufftDoubleComplex);

    DevBuf d_kern(bytes), d_imp(bytes), d_W(bytes),
           d_data(bytes), d_mask(bytes);

    CufftPlan plan(M, N);

    const int threads = 256;
    const int blocks  = (MN + threads - 1) / threads;
    const double inv_aR = 1.0 / aRange;

    // Build kernel and impulse in spatial domain
    build_kernel_g <<<blocks, threads>>>(d_kern.c(), M, N, cm, cn, inv_aR);
    build_impulse_g<<<blocks, threads>>>(d_imp.c(),  M, N, cm, cn);

    // FFT both (in-place)
    plan.fwd(d_kern.c());
    plan.fwd(d_imp.c());

    // W = FFT(kernel) / FFT(impulse) / (M*N)
    compute_W_g<<<blocks, threads>>>(d_W.c(), d_kern.c(), d_imp.c(),
                                     1.0 / (double)MN, MN);

    // Pad input data and mask into M×N complex arrays
    pad_data_g<<<blocks, threads>>>(d_data.c(), d_mask.c(), d_in, Nr, N, MN);

    // FFT data and mask
    plan.fwd(d_data.c());
    plan.fwd(d_mask.c());

    // Multiply by W
    cmul_g<<<blocks, threads>>>(d_data.c(), d_data.c(), d_W.c(), MN);
    cmul_g<<<blocks, threads>>>(d_mask.c(), d_mask.c(), d_W.c(), MN);

    // IFFT (unnormalized, matching R's fft(..., inverse=TRUE))
    plan.inv(d_data.c());
    plan.inv(d_mask.c());

    // Extract top-left Nr×Nr: result = Re(data) / Re(mask)
    dim3 block2d(16, 16);
    dim3 grid2d((Nr + 15) / 16, (Nr + 15) / 16);
    extract_g<<<grid2d, block2d>>>(d_out, d_data.c(), d_mask.c(), Nr, N, tol);

    chk_cuda(cudaDeviceSynchronize(), "image_smooth_gpu sync");
}

// ===========================================================================
// Standalone verifier — compiled only when -DSMOOTH_VERIFY_MAIN is set
// ===========================================================================
#ifdef SMOOTH_VERIFY_MAIN

#include "csv_io.cuh"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

int main(int argc, char* argv[])
{
    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s input.csv ref.csv output.csv\n\n"
            "  input.csv  — raw 60x60 matrix (from R write.csv)\n"
            "  ref.csv    — fields::image.smooth reference (from R write.csv)\n"
            "  output.csv — GPU result written here\n",
            argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];
    const int Nr = 60;

    // ---- Read input -------------------------------------------------------
    int nrows = 0, ncols = 0;
    std::vector<double> h_in;
    try {
        h_in = read_csv_flat(in_path, nrows, ncols, true);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading input '%s': %s\n", in_path, e.what());
        return 1;
    }
    if (nrows != Nr || ncols != Nr) {
        fprintf(stderr, "ERROR: expected %dx%d, got %dx%d\n", Nr, Nr, nrows, ncols);
        return 1;
    }
    printf("Input  : %s  (%dx%d)\n", in_path, nrows, ncols);

    // ---- GPU smooth -------------------------------------------------------
    double *d_in = nullptr, *d_out_dev = nullptr;
    size_t nbytes = (size_t)Nr * Nr * sizeof(double);

    cudaError_t cerr = cudaMalloc(&d_in, nbytes);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_in: %s\n", cudaGetErrorString(cerr));
        return 1;
    }
    cerr = cudaMalloc(&d_out_dev, nbytes);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out: %s\n", cudaGetErrorString(cerr));
        cudaFree(d_in); return 1;
    }

    cudaMemcpy(d_in, h_in.data(), nbytes, cudaMemcpyHostToDevice);

    try {
        image_smooth_gpu(d_in, d_out_dev, Nr);
    } catch (const std::exception& e) {
        fprintf(stderr, "GPU error: %s\n", e.what());
        cudaFree(d_in); cudaFree(d_out_dev); return 1;
    }

    std::vector<double> h_out(Nr * Nr);
    cudaMemcpy(h_out.data(), d_out_dev, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out_dev);

    // ---- Write GPU output -------------------------------------------------
    try {
        write_csv_flat(out_path, h_out.data(), Nr, Nr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR writing output '%s': %s\n", out_path, e.what());
        return 1;
    }
    printf("Written: %s\n", out_path);

    // ---- Read reference ---------------------------------------------------
    int ref_rows = 0, ref_cols = 0;
    std::vector<double> h_ref;
    try {
        h_ref = read_csv_flat(ref_path, ref_rows, ref_cols, true);
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
    int    total = Nr * Nr;
    int    nan_mismatch   = 0;
    int    cells_compared = 0;
    double max_diff       = 0.0;
    double sum_diff       = 0.0;
    int    gt_1e6         = 0;

    struct BadCell { int idx; double diff; };
    std::vector<BadCell> bad;
    bad.reserve(total);

    for (int i = 0; i < total; i++) {
        bool gn = std::isnan(h_out[i]);
        bool rn = std::isnan(h_ref[i]);
        if (gn || rn) {
            if (gn != rn) ++nan_mismatch;
            continue;
        }
        double d = std::fabs(h_out[i] - h_ref[i]);
        max_diff  = std::max(max_diff, d);
        sum_diff += d;
        ++cells_compared;
        if (d > 1e-6) ++gt_1e6;
        bad.push_back({i, d});
    }
    double mean_diff = (cells_compared > 0) ? sum_diff / cells_compared : 0.0;

    std::sort(bad.begin(), bad.end(),
              [](const BadCell& a, const BadCell& b){ return a.diff > b.diff; });

    printf("\n=== Image Smooth Verification ===\n");
    printf("Total cells        : %d\n",   total);
    printf("Cells compared     : %d\n",   cells_compared);
    printf("NaN mismatches     : %d\n",   nan_mismatch);
    printf("Max  |GPU - R|     : %.3e\n", max_diff);
    printf("Mean |GPU - R|     : %.3e\n", mean_diff);
    printf("Cells > 1e-6       : %d\n",   gt_1e6);

    int show = (int)std::min((size_t)5, bad.size());
    if (show > 0 && bad[0].diff > 1e-10) {
        printf("\nTop-%d worst-diff cells (row col GPU_value R_value diff):\n", show);
        for (int i = 0; i < show; i++) {
            int row = bad[i].idx / Nr, col = bad[i].idx % Nr;
            printf("  [%2d,%2d]  GPU=%.10g  R=%.10g  |diff|=%.3e\n",
                   row, col, h_out[bad[i].idx], h_ref[bad[i].idx], bad[i].diff);
        }
    }

    bool pass = (nan_mismatch == 0) && (max_diff < 1e-6);
    printf("\nResult : %s\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        if (nan_mismatch > 0)
            printf("  Reason: %d NaN/NA mismatches\n", nan_mismatch);
        if (max_diff >= 1e-6)
            printf("  Reason: max |diff| = %.3e >= 1e-6\n", max_diff);
    }
    return pass ? 0 : 1;
}

#endif // SMOOTH_VERIFY_MAIN
