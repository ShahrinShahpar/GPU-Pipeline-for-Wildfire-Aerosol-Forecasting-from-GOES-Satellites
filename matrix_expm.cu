// matrix_expm.cu — GPU matrix exponential, Padé degree-13 + scaling-and-squaring.
//
// Matches R's expm::expm(G) (Higham08 algorithm).
// Reference: Higham "Functions of Matrices: Theory and Computation" (2008),
//            Algorithm 10.20.
//
// Build as standalone verifier:
//   nvcc -arch=sm_70 -std=c++14 -O3 -I. -DEXPM_VERIFY_MAIN \
//        matrix_expm.cu csv_io.cu -o matrix_expm_verify \
//        -lcublas -lcusolver
//
// Usage:
//   ./matrix_expm_verify input_G.csv ref_expG.csv [out_expG.csv]

#include "matrix_expm.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Padé-13 coefficients (Higham 2008, Table 10.4)
// ---------------------------------------------------------------------------
static const double PADE_B[14] = {
    64764752532480000.0,
    32382376266240000.0,
     7771770303897600.0,
     1187353796428800.0,
      129060195264000.0,
       10559470521600.0,
         670442572800.0,
          33522128640.0,
            1323241920.0,
              40840800.0,
                960960.0,
                 16380.0,
                   182.0,
                     1.0
};
static const double THETA_13 = 5.371920351148152;

// ---------------------------------------------------------------------------
// Error-check helpers
// ---------------------------------------------------------------------------
static void chk(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
}
static void chk_blas(cublasStatus_t e, const char* w) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(w) + " cublas=" + std::to_string((int)e));
}
static void chk_sol(cusolverStatus_t e, const char* w) {
    if (e != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string(w) + " cusolver=" + std::to_string((int)e));
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Scale matrix: A *= alpha (used before squaring and after)
static __global__ void scale_matrix(double* A, double alpha, int NN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NN) A[i] *= alpha;
}

// C = alpha * A + beta * B  (element-wise, all N×N)
static __global__ void axpby(double* C, const double* A, const double* B,
                              double alpha, double beta, int NN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NN) C[i] = alpha * A[i] + beta * B[i];
}

// Add scalar to diagonal: A[i,i] += val
static __global__ void add_diag(double* A, double val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) A[i * N + i] += val;
}

// Row-wise sum of absolute values to compute ||A||_inf = max_row sum_j |A[i,j]|
// One block per row, uses shared memory reduction
static __global__ void row_abs_sum(const double* A, double* row_sums, int N) {
    extern __shared__ double smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    double s = 0.0;
    for (int j = tid; j < N; j += blockDim.x)
        s += fabs(A[row * N + j]);
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) row_sums[row] = smem[0];
}

// ---------------------------------------------------------------------------
// RAII device buffer
// ---------------------------------------------------------------------------
struct DevBuf {
    double* ptr = nullptr;
    int     n   = 0;
    DevBuf() = default;
    DevBuf(int n_) : n(n_) { chk(cudaMalloc(&ptr, n_ * sizeof(double)), "DevBuf alloc"); }
    ~DevBuf() { if (ptr) cudaFree(ptr); }
    // Non-copyable
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

// ---------------------------------------------------------------------------
// Helper: row-major matrix multiply C = A * B using cuBLAS
// cuBLAS is column-major: passing row-major A,B makes it compute (A*B) correctly
// when we swap the argument order: dgemm(B, A) → column-major B^T * A^T = (A*B)^T
// Storing that as row-major gives A*B.
// ---------------------------------------------------------------------------
static void mm(cublasHandle_t blas, int N,
               const double* d_A, const double* d_B, double* d_C,
               double alpha = 1.0, double beta = 0.0)
{
    chk_blas(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, N, N, &alpha,
                         d_B, N, d_A, N, &beta, d_C, N),
             "dgemm");
}

// ---------------------------------------------------------------------------
// Compute infinity norm: max_i sum_j |A[i,j]|
// ---------------------------------------------------------------------------
static double inf_norm(const double* d_A, int N, double* d_scratch) {
    const int T = 128;
    row_abs_sum<<<N, T, T * sizeof(double)>>>(d_A, d_scratch, N);
    chk(cudaDeviceSynchronize(), "inf_norm sync");
    auto it = thrust::max_element(thrust::device_ptr<double>(d_scratch),
                                  thrust::device_ptr<double>(d_scratch + N));
    double val;
    chk(cudaMemcpy(&val, thrust::raw_pointer_cast(it), sizeof(double),
                   cudaMemcpyDeviceToHost), "inf_norm copy");
    return val;
}

// ---------------------------------------------------------------------------
// matrix_expm_gpu
// ---------------------------------------------------------------------------
void matrix_expm_gpu(const double* d_A_in, double* d_out, int N)
{
    const int NN = N * N;
    const int T  = 256;
    const int BL = (NN + T - 1) / T;

    cublasHandle_t    blas;
    cusolverDnHandle_t sol;
    chk_blas(cublasCreate(&blas),    "cublasCreate");
    chk_sol (cusolverDnCreate(&sol), "cusolverCreate");

    // Working copy of A (will be scaled)
    DevBuf d_A(NN);
    chk(cudaMemcpy(d_A.ptr, d_A_in, NN * sizeof(double), cudaMemcpyDeviceToDevice),
        "copy A");

    // Scratch for inf-norm
    DevBuf d_rnorm(N);

    // 1. Compute s (scaling exponent)
    double norm = inf_norm(d_A.ptr, N, d_rnorm.ptr);
    double s_d  = std::max(0.0, std::ceil(std::log2(norm / THETA_13)));
    int    s    = (int)s_d;

    // Scale A ← A / 2^s
    if (s > 0) {
        double inv_scale = 1.0 / std::pow(2.0, (double)s);
        scale_matrix<<<BL,T>>>(d_A.ptr, inv_scale, NN);
        chk(cudaDeviceSynchronize(), "scale");
    }

    // 2. A2 = A^2,  A4 = A2^2,  A6 = A2 * A4
    DevBuf A2(NN), A4(NN), A6(NN);
    mm(blas, N, d_A.ptr, d_A.ptr, A2.ptr);          // A2 = A * A
    mm(blas, N, A2.ptr,  A2.ptr,  A4.ptr);          // A4 = A2 * A2
    mm(blas, N, A2.ptr,  A4.ptr,  A6.ptr);          // A6 = A2 * A4 (= A^6)

    // 3. Compute U and V (Padé numerator/denominator factors)
    //
    //   W1  = b[13]*A6 + b[11]*A4 + b[ 9]*A2
    //   W2  = b[ 7]*A6 + b[ 5]*A4 + b[ 3]*A2 + b[1]*I
    //   Z1  = b[12]*A6 + b[10]*A4 + b[ 8]*A2
    //   Z2  = b[ 6]*A6 + b[ 4]*A4 + b[ 2]*A2 + b[0]*I
    //
    //   W   = A6 * W1 + W2     →  then  U = A * W
    //   V   = A6 * Z1 + Z2

    DevBuf W1(NN), W2(NN), Z1(NN), Z2(NN);
    DevBuf W(NN), U(NN), V(NN);

    // W1 = b13*A6 + b11*A4 + b9*A2
    axpby<<<BL,T>>>(W1.ptr, A6.ptr, A4.ptr, PADE_B[13], PADE_B[11], NN);
    chk(cudaDeviceSynchronize(), "W1 ab");
    axpby<<<BL,T>>>(W1.ptr, W1.ptr, A2.ptr, 1.0, PADE_B[9], NN);
    chk(cudaDeviceSynchronize(), "W1 a2");

    // W2 = b7*A6 + b5*A4 + b3*A2 + b1*I
    axpby<<<BL,T>>>(W2.ptr, A6.ptr, A4.ptr, PADE_B[7], PADE_B[5], NN);
    chk(cudaDeviceSynchronize(), "W2 ab");
    axpby<<<BL,T>>>(W2.ptr, W2.ptr, A2.ptr, 1.0, PADE_B[3], NN);
    chk(cudaDeviceSynchronize(), "W2 a2");
    add_diag<<<(N+T-1)/T, T>>>(W2.ptr, PADE_B[1], N);
    chk(cudaDeviceSynchronize(), "W2 diag");

    // Z1 = b12*A6 + b10*A4 + b8*A2
    axpby<<<BL,T>>>(Z1.ptr, A6.ptr, A4.ptr, PADE_B[12], PADE_B[10], NN);
    chk(cudaDeviceSynchronize(), "Z1 ab");
    axpby<<<BL,T>>>(Z1.ptr, Z1.ptr, A2.ptr, 1.0, PADE_B[8], NN);
    chk(cudaDeviceSynchronize(), "Z1 a2");

    // Z2 = b6*A6 + b4*A4 + b2*A2 + b0*I
    axpby<<<BL,T>>>(Z2.ptr, A6.ptr, A4.ptr, PADE_B[6], PADE_B[4], NN);
    chk(cudaDeviceSynchronize(), "Z2 ab");
    axpby<<<BL,T>>>(Z2.ptr, Z2.ptr, A2.ptr, 1.0, PADE_B[2], NN);
    chk(cudaDeviceSynchronize(), "Z2 a2");
    add_diag<<<(N+T-1)/T, T>>>(Z2.ptr, PADE_B[0], N);
    chk(cudaDeviceSynchronize(), "Z2 diag");

    // W = A6 * W1 + W2
    mm(blas, N, A6.ptr, W1.ptr, W.ptr);             // W = A6 * W1
    axpby<<<BL,T>>>(W.ptr, W.ptr, W2.ptr, 1.0, 1.0, NN);
    chk(cudaDeviceSynchronize(), "W");

    // U = A * W
    mm(blas, N, d_A.ptr, W.ptr, U.ptr);             // U = A * W

    // V = A6 * Z1 + Z2
    mm(blas, N, A6.ptr, Z1.ptr, V.ptr);             // V = A6 * Z1
    axpby<<<BL,T>>>(V.ptr, V.ptr, Z2.ptr, 1.0, 1.0, NN);
    chk(cudaDeviceSynchronize(), "V");

    // 4. P = U + V  (numerator),  Q = -U + V  (denominator)
    //    Write P into d_out (reuse as scratch), Q into d_A (no longer needed as A)
    axpby<<<BL,T>>>(d_out,    U.ptr, V.ptr,  1.0,  1.0, NN);   // P = U + V
    axpby<<<BL,T>>>(d_A.ptr,  U.ptr, V.ptr, -1.0,  1.0, NN);   // Q = -U + V
    chk(cudaDeviceSynchronize(), "PQ");

    // 5. Solve Q * F = P  (cuSOLVER, column-major convention)
    //    Passing row-major Q as col-major gives Q^T; solving Q^T * X = P^T yields F^T
    //    as col-major = F as row-major (valid because P and Q commute as polynomials in A)
    DevBuf d_ipiv(N);     // actually int, but we'll use a separate int array
    int*   d_ipiv_int = nullptr;
    chk(cudaMalloc(&d_ipiv_int, N * sizeof(int)), "ipiv alloc");

    int lwork = 0;
    chk_sol(cusolverDnDgetrf_bufferSize(sol, N, N, d_A.ptr, N, &lwork),
            "getrf bufsize");
    DevBuf d_work(lwork > 0 ? lwork : 1);
    int*   d_info = nullptr;
    chk(cudaMalloc(&d_info, sizeof(int)), "info alloc");

    // LU factorize Q (in d_A.ptr)
    chk_sol(cusolverDnDgetrf(sol, N, N, d_A.ptr, N, d_work.ptr, d_ipiv_int, d_info),
            "getrf");
    chk(cudaDeviceSynchronize(), "getrf sync");
    {
        int h_info = 0;
        chk(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
            "getrf info copy");
        if (h_info != 0)
            throw std::runtime_error("getrf: singular matrix (info=" +
                                     std::to_string(h_info) + ")");
    }

    // Solve Q * F = P  (d_out holds P, will be overwritten with F)
    chk_sol(cusolverDnDgetrs(sol, CUBLAS_OP_N, N, N,
                             d_A.ptr, N, d_ipiv_int,
                             d_out,   N, d_info),
            "getrs");
    chk(cudaDeviceSynchronize(), "getrs sync");

    cudaFree(d_ipiv_int);
    cudaFree(d_info);

    // 6. Square s times: F ← F^2  (s squarings)
    if (s > 0) {
        DevBuf d_tmp(NN);
        for (int i = 0; i < s; i++) {
            mm(blas, N, d_out, d_out, d_tmp.ptr);   // tmp = F * F
            chk(cudaMemcpy(d_out, d_tmp.ptr, NN * sizeof(double),
                           cudaMemcpyDeviceToDevice), "squaring copy");
        }
    }

    cusolverDnDestroy(sol);
    cublasDestroy(blas);
}

// ===========================================================================
// Standalone verifier — compiled only when -DEXPM_VERIFY_MAIN is set
// ===========================================================================
#ifdef EXPM_VERIFY_MAIN

#include "csv_io.cuh"
#include <cstdio>
#include <algorithm>

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s input_G.csv ref_expG.csv [out_expG.csv]\n\n"
            "  input_G.csv  — GPU-computed G matrix (400×400)\n"
            "  ref_expG.csv — R's expm(G) reference\n"
            "  out_expG.csv — optional: write GPU expm(G) here\n",
            argv[0]);
        return 1;
    }
    const char* in_path  = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = (argc >= 4) ? argv[3] : nullptr;

    // ---- Read input -------------------------------------------------------
    int nrows = 0, ncols = 0;
    std::vector<double> h_in;
    try {
        h_in = read_csv_flat(in_path, nrows, ncols, true);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading input '%s': %s\n", in_path, e.what());
        return 1;
    }
    if (nrows != ncols) {
        fprintf(stderr, "ERROR: input must be square, got %dx%d\n", nrows, ncols);
        return 1;
    }
    int N  = nrows;
    int NN = N * N;
    printf("Input  : %s  (%d×%d)\n", in_path, N, N);

    // ---- GPU expm ---------------------------------------------------------
    double *d_in = nullptr, *d_out = nullptr;
    chk(cudaMalloc(&d_in,  NN * sizeof(double)), "alloc d_in");
    chk(cudaMalloc(&d_out, NN * sizeof(double)), "alloc d_out");
    chk(cudaMemcpy(d_in, h_in.data(), NN * sizeof(double),
                   cudaMemcpyHostToDevice), "H2D");

    try {
        matrix_expm_gpu(d_in, d_out, N);
    } catch (const std::exception& e) {
        fprintf(stderr, "GPU error: %s\n", e.what());
        cudaFree(d_in); cudaFree(d_out);
        return 1;
    }

    std::vector<double> h_out(NN);
    chk(cudaMemcpy(h_out.data(), d_out, NN * sizeof(double),
                   cudaMemcpyDeviceToHost), "D2H");
    cudaFree(d_in);
    cudaFree(d_out);

    printf("expm[0,0]=%.10g  expm[220,220]=%.10g\n",
           h_out[0], h_out[220*N+220]);

    // ---- Optionally write output ------------------------------------------
    if (out_path) {
        try {
            write_csv_flat(out_path, h_out.data(), N, N);
            printf("Written: %s\n", out_path);
        } catch (const std::exception& e) {
            fprintf(stderr, "ERROR writing output: %s\n", e.what());
        }
    }

    // ---- Read reference ---------------------------------------------------
    int rr = 0, rc = 0;
    std::vector<double> h_ref;
    try {
        h_ref = read_csv_flat(ref_path, rr, rc, true);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR reading reference '%s': %s\n", ref_path, e.what());
        return 1;
    }
    if (rr != N || rc != N) {
        fprintf(stderr, "ERROR: reference expected %d×%d, got %d×%d\n", N, N, rr, rc);
        return 1;
    }

    // ---- Field-by-field comparison ----------------------------------------
    int    total    = NN;
    int    compared = 0;
    double max_diff = 0.0, sum_diff = 0.0;
    int    gt_1e6   = 0;

    struct BadCell { int idx; double diff, gpu, ref; };
    std::vector<BadCell> bad;
    bad.reserve(total);

    for (int i = 0; i < total; i++) {
        bool gn = std::isnan(h_out[i]);
        bool rn = std::isnan(h_ref[i]);
        if (gn || rn) continue;
        double d = std::fabs(h_out[i] - h_ref[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
        ++compared;
        if (d > 1e-6) ++gt_1e6;
        bad.push_back({i, d, h_out[i], h_ref[i]});
    }
    double mean_diff = compared > 0 ? sum_diff / compared : 0.0;

    std::sort(bad.begin(), bad.end(),
              [](const BadCell& a, const BadCell& b){ return a.diff > b.diff; });

    printf("\n=== Matrix Expm Verification ===\n");
    printf("Matrix size        : %d × %d\n", N, N);
    printf("Cells compared     : %d\n",   compared);
    printf("Max  |GPU - R|     : %.3e\n", max_diff);
    printf("Mean |GPU - R|     : %.3e\n", mean_diff);
    printf("Cells > 1e-6       : %d\n",   gt_1e6);

    int show = (int)std::min((size_t)5, bad.size());
    if (show > 0 && bad[0].diff > 1e-10) {
        printf("\nTop-%d worst cells (row col GPU R diff):\n", show);
        for (int i = 0; i < show; i++) {
            int r = bad[i].idx / N, c = bad[i].idx % N;
            printf("  [%3d,%3d]  GPU=%.10g  R=%.10g  |diff|=%.3e\n",
                   r, c, bad[i].gpu, bad[i].ref, bad[i].diff);
        }
    }

    bool pass = (max_diff < 1e-6);
    printf("\nResult : %s\n", pass ? "PASS" : "FAIL");
    if (!pass)
        printf("  Reason: max |diff| = %.3e >= 1e-6\n", max_diff);
    return pass ? 0 : 1;
}

#endif // EXPM_VERIFY_MAIN
