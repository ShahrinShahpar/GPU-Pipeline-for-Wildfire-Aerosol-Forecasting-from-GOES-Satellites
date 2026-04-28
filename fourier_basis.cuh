#pragma once
// GPU Fourier basis — Function_Omega + Function_F from util.R
// Row-major storage: d_F[site * n_cols + col]
// Compatible with obs_assembly_gpu (which expects row-major d_F)

#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generate Omega1 (4 rows) and Omega2 (n2 rows) frequency grids.
// Replicates R's Function_Omega(N) exactly.
// o{1,2}_k1[j], o{1,2}_k2[j] are interleaved as separate vectors for easy upload.
void generate_omega(int N,
                    std::vector<double>& o1_k1, std::vector<double>& o1_k2,
                    std::vector<double>& o2_k1, std::vector<double>& o2_k2);

// On-device Fourier basis matrix F (row-major).
// F[site, col] = d_F[site * n_cols + col]
struct FourierBasisGPU {
    double* d_F = nullptr;  // [n_sites * n_cols] row-major
    int n_sites = 0;        // Nr * Nr
    int n_cols  = 0;        // N^2
    int Nr = 0, N = 0;
    int n1 = 0, n2 = 0;    // |Omega1|, |Omega2|

    // Compute F entirely on GPU (F stays on device).
    void build(int Nr_, int N_);

    ~FourierBasisGPU();
};
