// fourier_basis.cu — pipeline Fourier basis with row-major storage.
// Adapted from Physics-Informed.../fourier_basis_gpu.cu (CPU verify/main removed).
//
// Key difference from original: F stored ROW-MAJOR so obs_assembly_gpu
// can consume it directly without transposition.

#include "fourier_basis.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define FB_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "CUDA error [fourier_basis] " << __FILE__ << ":" \
                  << __LINE__ << " — " << cudaGetErrorString(_e) << std::endl; \
        exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// generate_omega — CPU, exact replica of R's Function_Omega(N)
// ---------------------------------------------------------------------------
void generate_omega(int N,
                    std::vector<double>& o1_k1, std::vector<double>& o1_k2,
                    std::vector<double>& o2_k1, std::vector<double>& o2_k2)
{
    int half = N / 2;

    // Omega1: four corners
    o1_k1 = {0.0, 0.0, (double)half, (double)half};
    o1_k2 = {0.0, (double)half, 0.0, (double)half};

    // Omega2_p1
    for (int k1 = 1; k1 < half; k1++) {
        o2_k1.push_back(k1);
        o2_k2.push_back(half);
    }
    // Omega2_p2: expand.grid(k1=0..N/2, k2=1..N/2-1) — k1 varies fastest
    for (int k2 = 1; k2 < half; k2++) {
        for (int k1 = 0; k1 <= half; k1++) {
            o2_k1.push_back(k1);
            o2_k2.push_back(k2);
        }
    }
    // Omega2_p3: expand.grid(k1=1..N/2-1, k2=-N/2+1..0) — k1 varies fastest
    for (int k2 = -half + 1; k2 <= 0; k2++) {
        for (int k1 = 1; k1 < half; k1++) {
            o2_k1.push_back(k1);
            o2_k2.push_back(k2);
        }
    }
}

// ---------------------------------------------------------------------------
// compute_F_kernel_rm — one thread per (site, col) entry, row-major output
// F[site * n_total + col] = trig(2π x k1 + 2π y k2)
// ---------------------------------------------------------------------------
__global__ void compute_F_kernel_rm(
    double* __restrict__ d_F,
    int Nr, int n_sites,
    int n1, int n2,
    const double* __restrict__ d_o1_k1, const double* __restrict__ d_o1_k2,
    const double* __restrict__ d_o2_k1, const double* __restrict__ d_o2_k2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_total = n1 + 2 * n2;
    if (idx >= n_sites * n_total) return;

    int site = idx / n_total;   // which spatial site
    int col  = idx % n_total;   // which Fourier basis column

    // Spatial coordinates (expand.grid: x varies fastest)
    double x = (site % Nr) / (double)Nr;
    double y = (site / Nr) / (double)Nr;

    double val;
    if (col < n1) {
        double angle = 2.0 * M_PI * (x * d_o1_k1[col] + y * d_o1_k2[col]);
        val = cos(angle);
    } else if (col < n1 + n2) {
        int j = col - n1;
        double angle = 2.0 * M_PI * (x * d_o2_k1[j] + y * d_o2_k2[j]);
        val = 2.0 * cos(angle);
    } else {
        int j = col - n1 - n2;
        double angle = 2.0 * M_PI * (x * d_o2_k1[j] + y * d_o2_k2[j]);
        val = 2.0 * sin(angle);
    }

    // Row-major: F[site, col]
    d_F[site * n_total + col] = val;
}

// ---------------------------------------------------------------------------
// FourierBasisGPU::build
// ---------------------------------------------------------------------------
void FourierBasisGPU::build(int Nr_, int N_)
{
    Nr = Nr_; N = N_;
    n_sites = Nr * Nr;

    std::vector<double> o1_k1, o1_k2, o2_k1, o2_k2;
    generate_omega(N, o1_k1, o1_k2, o2_k1, o2_k2);
    n1 = (int)o1_k1.size();
    n2 = (int)o2_k1.size();
    n_cols = n1 + 2 * n2;

    std::cout << "[fourier_basis] Nr=" << Nr << "  N=" << N
              << "  |Omega1|=" << n1 << "  |Omega2|=" << n2
              << "  F: " << n_sites << " x " << n_cols << std::endl;

    double *d_o1k1, *d_o1k2, *d_o2k1, *d_o2k2;
    FB_CUDA_CHECK(cudaMalloc(&d_o1k1, n1 * sizeof(double)));
    FB_CUDA_CHECK(cudaMalloc(&d_o1k2, n1 * sizeof(double)));
    FB_CUDA_CHECK(cudaMalloc(&d_o2k1, n2 * sizeof(double)));
    FB_CUDA_CHECK(cudaMalloc(&d_o2k2, n2 * sizeof(double)));
    FB_CUDA_CHECK(cudaMemcpy(d_o1k1, o1_k1.data(), n1*sizeof(double), cudaMemcpyHostToDevice));
    FB_CUDA_CHECK(cudaMemcpy(d_o1k2, o1_k2.data(), n1*sizeof(double), cudaMemcpyHostToDevice));
    FB_CUDA_CHECK(cudaMemcpy(d_o2k1, o2_k1.data(), n2*sizeof(double), cudaMemcpyHostToDevice));
    FB_CUDA_CHECK(cudaMemcpy(d_o2k2, o2_k2.data(), n2*sizeof(double), cudaMemcpyHostToDevice));

    FB_CUDA_CHECK(cudaMalloc(&d_F, (size_t)n_sites * n_cols * sizeof(double)));

    int total = n_sites * n_cols;
    int block = 256, grid = (total + 255) / 256;
    compute_F_kernel_rm<<<grid, block>>>(
        d_F, Nr, n_sites, n1, n2,
        d_o1k1, d_o1k2, d_o2k1, d_o2k2);
    FB_CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_o1k1); cudaFree(d_o1k2);
    cudaFree(d_o2k1); cudaFree(d_o2k2);
    std::cout << "[fourier_basis] F built on GPU ("
              << (size_t)n_sites * n_cols * 8 / 1024.0 / 1024.0 << " MB)" << std::endl;
}

FourierBasisGPU::~FourierBasisGPU()
{
    if (d_F) { cudaFree(d_F); d_F = nullptr; }
}
