// gibbs_sampler.cu — GPU Gibbs FFBS2 sampler, pipeline version.
// Adapted from Physics-Informed.../gibbs_step6.cu:
//   - main() and file-reading removed
//   - GPUGibbsSampler renamed GibbsSamplerPipeline
//   - getMflt() method added
//   - run_gibbs_pipeline() replaces Gibbs_FFBS2_Step6()

#include "gibbs_sampler.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

using namespace std;

#define GS_CUDA_CHECK(call) do { \
    cudaError_t _e = call; \
    if (_e != cudaSuccess) { \
        cerr << "CUDA error [gibbs] " << __FILE__ << ":" << __LINE__ \
             << " - " << cudaGetErrorString(_e) << endl; exit(1); } \
} while(0)
#define GS_CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = call; \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        cerr << "cuBLAS error [gibbs] " << __FILE__ << ":" << __LINE__ << endl; exit(1); } \
} while(0)
#define GS_CUSOLVER_CHECK(call) do { \
    cusolverStatus_t _s = call; \
    if (_s != CUSOLVER_STATUS_SUCCESS) { \
        cerr << "cuSOLVER error [gibbs] " << __FILE__ << ":" << __LINE__ \
             << " code=" << _s << endl; exit(1); } \
} while(0)
#define GS_CURAND_CHECK(call) do { \
    curandStatus_t _s = call; \
    if (_s != CURAND_STATUS_SUCCESS) { \
        cerr << "cuRAND error [gibbs] " << __FILE__ << ":" << __LINE__ \
             << " code=" << _s << endl; exit(1); } \
} while(0)

// ---------------------------------------------------------------------------
// CPU Wishart sampler — matches R's rwish(nu, Phi) exactly in distribution.
//
// Algorithm: Wishart(nu, Phi) = Phi^{1/2} * Z * Z^T * Phi^{1/2}
//   where Z is (p × nu) with i.i.d. N(0,1) entries.
// Since Phi = phi_scale * I (diagonal), Phi^{1/2} = sqrt(phi_scale) * I, so:
//   W = phi_scale * Z * Z^T
// This gives E[W] = nu * Phi (matching the Wishart mean).
//
// Cost: p*nu normal samples + one p×p symmetric matrix multiply via Eigen BLAS.
// For p=nu=800 this is ~640k normal samples and ~512M FLOPs (~1-2s on CPU).
// ---------------------------------------------------------------------------
static MatrixXd rwish_cpu(mt19937& rng, int nu, const MatrixXd& Phi) {
    int p = Phi.rows();
    double phi_scale = Phi(0, 0);   // assumes Phi = phi_scale * I
    normal_distribution<double> nd(0.0, 1.0);
    MatrixXd Z(p, nu);
    for (int j = 0; j < nu; j++)
        for (int i = 0; i < p; i++)
            Z(i, j) = nd(rng);
    // selfadjointView gives symmetric rank-update: W = phi_scale * Z * Z^T
    MatrixXd W(p, p);
    W.setZero();
    W.selfadjointView<Eigen::Lower>().rankUpdate(Z, phi_scale);
    W = W.selfadjointView<Eigen::Lower>();
    return W;
}

// ---------------------------------------------------------------------------
// Small GPU helpers
// ---------------------------------------------------------------------------
__global__ void gs_add_diag_reg(double* M, int n, double reg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) M[i*n + i] += reg;
}

__global__ void gs_squared_norm(const double* v, double* out, int n) {
    __shared__ double sh[256];
    int tid = threadIdx.x, idx = blockIdx.x*blockDim.x + tid;
    sh[tid] = (idx < n) ? v[idx]*v[idx] : 0.0;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid+s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sh[0]);
}

// ---------------------------------------------------------------------------
// CPU sampling helpers
// ---------------------------------------------------------------------------
static double rinvgamma(mt19937& gen, double alpha, double beta) {
    gamma_distribution<double> gd(alpha, 1.0/beta);
    return 1.0 / gd(gen);
}

static VectorXd rmvnorm(mt19937& gen, const VectorXd& mean, const MatrixXd& cov) {
    int n = mean.size();
    normal_distribution<double> nd(0.0, 1.0);
    Eigen::LLT<MatrixXd> llt(cov);
    MatrixXd L = llt.matrixL();
    VectorXd z(n);
    for (int i = 0; i < n; i++) z(i) = nd(gen);
    return mean + L * z;
}

// Sample from Inverse-Wishart(nu, Phi) matching R's MCMCpack::riwish(nu, Phi).
//
// Algorithm avoids the numerically unstable dense matrix inverse W.inverse():
//   1. Generate Bartlett lower-triangular A:
//        A_ii ~ chi(nu-i),  A_ij ~ N(0,1) for j<i
//      so that (L @ A) @ (L @ A)^T ~ Wishart(nu, Phi)  where L = chol(Phi)
//   2. Set Z = L^{-1} @ A^{-1}  (two triangular solves — no dense inverse)
//   3. Return W_IW = Z^T @ Z  (symmetric rank-update via Eigen dsyrk)
//
// This is O(p^3/3) triangular solves instead of O(p^3) dense LU inverse,
// and is numerically stable for large p (here p = 800).
static MatrixXd riwish(mt19937& gen, int nu, const MatrixXd& Phi) {
    int p = Phi.rows();
    // --- Step 1: build Bartlett lower-triangular A ---
    MatrixXd A = MatrixXd::Zero(p, p);
    normal_distribution<double> nd(0.0, 1.0);
    // Use chi(nu) for all diagonal entries — matches R MCMCpack::riwish and gibbs_step6.cu.
    // (R's rchisq(p, v:(v-p+1)) produces decreasing degrees, but MCMCpack uses the same
    // nu for all, which is what the original GPU implementation matched.)
    chi_squared_distribution<double> cs(nu);
    for (int i = 0; i < p; i++) {
        A(i, i) = sqrt(cs(gen));
        for (int j = 0; j < i; j++) A(i, j) = nd(gen);
    }
    // --- Step 2: Z = L^{-1} @ A^{-1}  (two triangular solves) ---
    // (a) Solve A @ X = I  →  X = A^{-1}  (lower-triangular solve)
    MatrixXd X = A.triangularView<Eigen::Lower>()
                  .solve(MatrixXd::Identity(p, p));
    // (b) Solve L @ Z = X  →  Z = L^{-1} @ X  (lower-triangular solve)
    // llt.matrixL() is already a TriangularView<Lower>, .solve() works directly
    Eigen::LLT<MatrixXd> llt(Phi);
    MatrixXd Z = llt.matrixL().solve(X);
    // --- Step 3: W_IW = Z^T @ Z  (symmetric rank-update) ---
    MatrixXd W(p, p);
    W.setZero();
    W.selfadjointView<Eigen::Lower>().rankUpdate(Z.transpose());
    return W.selfadjointView<Eigen::Lower>();
}

// ---------------------------------------------------------------------------
// GibbsSamplerPipeline — GPU Kalman filter + FFBS on device
// ---------------------------------------------------------------------------
class GibbsSamplerPipeline {
public:
    cublasHandle_t   cublas;
    cusolverDnHandle_t solver;
    curandGenerator_t curand_gen;

    int n_state, max_obs, max_T;

    double *d_G, *d_W;
    double *d_m_flt, *d_C_flt, *d_m_prd, *d_C_prd;
    double *d_Ftt, *d_yt, *d_Vt;
    double *d_Q, *d_K_gain, *d_innov;
    double *d_theta, *d_ht, *d_Ht, *d_L_Ht, *d_z;
    double *d_F1t, *d_F2t, *d_y1, *d_y2, *d_residual, *d_rsd_sq;
    double *d_state_rsd;
    double *d_tmp1, *d_tmp2, *d_tmp3, *d_tmp4, *d_tmp5;
    int    *d_info, *d_ipiv;
    double *d_work;
    int lwork;

    vector<double*> d_m_flt_all, d_C_flt_all, d_m_prd_all, d_C_prd_all;

    GibbsSamplerPipeline(int n_state_, int max_obs_, int max_T_, unsigned long long curand_seed=42)
        : n_state(n_state_), max_obs(max_obs_), max_T(max_T_)
    {
        GS_CUBLAS_CHECK(cublasCreate(&cublas));
        GS_CUSOLVER_CHECK(cusolverDnCreate(&solver));
        GS_CURAND_CHECK(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        GS_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_gen, curand_seed));

        auto alloc = [&](double** p, size_t n){ GS_CUDA_CHECK(cudaMalloc(p, n*sizeof(double))); };
        alloc(&d_G, (size_t)n_state*n_state);
        alloc(&d_W, (size_t)n_state*n_state);
        alloc(&d_m_flt, n_state);
        alloc(&d_C_flt, (size_t)n_state*n_state);
        alloc(&d_m_prd, n_state);
        alloc(&d_C_prd, (size_t)n_state*n_state);
        alloc(&d_Ftt,   (size_t)max_obs*n_state);
        alloc(&d_yt,    max_obs);
        alloc(&d_Vt,    (size_t)max_obs*max_obs);
        alloc(&d_Q,     (size_t)max_obs*max_obs);
        alloc(&d_K_gain,(size_t)n_state*max_obs);
        // mobs: backward pass reuses obs-sized buffers with n_state-sized data
        int mobs = max(max_obs, n_state);
        alloc(&d_innov, mobs);
        alloc(&d_theta, (size_t)max_T*n_state);
        alloc(&d_ht,    n_state);
        alloc(&d_Ht,    (size_t)n_state*n_state);
        alloc(&d_L_Ht,  (size_t)n_state*n_state);
        alloc(&d_z,     n_state);
        int half = n_state/2;
        alloc(&d_F1t,   (size_t)mobs*half);
        alloc(&d_F2t,   (size_t)mobs*half);
        alloc(&d_y1,    mobs);
        alloc(&d_y2,    mobs);
        alloc(&d_residual, mobs);
        alloc(&d_rsd_sq,   1);
        alloc(&d_state_rsd,(size_t)n_state*(max_T>1?max_T-1:1));
        size_t wmax = max((size_t)n_state*n_state, (size_t)mobs*(size_t)mobs);
        alloc(&d_tmp1, wmax);
        alloc(&d_tmp2, wmax);
        alloc(&d_tmp3, wmax);          // must hold both n_obs×n_obs and n_state×n_state
        alloc(&d_tmp4, (size_t)n_state*mobs);  // used as n_state×n_obs in filter AND n_state×n_state in backward
        alloc(&d_tmp5, (size_t)n_state*n_state);
        GS_CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        GS_CUDA_CHECK(cudaMalloc(&d_ipiv, max(max_obs,n_state)*sizeof(int)));

        // Workspace must cover: getrf(max_obs×max_obs), getrf(n_state×n_state), potrf(n_state×n_state)
        GS_CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(solver, max_obs, max_obs, d_Q,   max_obs, &lwork));
        int lw2, lw3;
        GS_CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(solver, n_state, n_state, d_tmp3, n_state, &lw2));
        GS_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, n_state, d_Ht, n_state, &lw3));
        lwork = max({lwork, lw2, lw3});
        GS_CUDA_CHECK(cudaMalloc(&d_work, lwork*sizeof(double)));

        for (int t = 0; t <= max_T; t++) {
            double *pm, *pC, *pa, *pb;
            GS_CUDA_CHECK(cudaMalloc(&pm, n_state*sizeof(double)));
            GS_CUDA_CHECK(cudaMalloc(&pC, (size_t)n_state*n_state*sizeof(double)));
            GS_CUDA_CHECK(cudaMalloc(&pa, n_state*sizeof(double)));
            GS_CUDA_CHECK(cudaMalloc(&pb, (size_t)n_state*n_state*sizeof(double)));
            d_m_flt_all.push_back(pm);
            d_C_flt_all.push_back(pC);
            d_m_prd_all.push_back(pa);
            d_C_prd_all.push_back(pb);
        }
    }

    ~GibbsSamplerPipeline() {
        cudaFree(d_G); cudaFree(d_W);
        cudaFree(d_m_flt); cudaFree(d_C_flt); cudaFree(d_m_prd); cudaFree(d_C_prd);
        cudaFree(d_Ftt); cudaFree(d_yt); cudaFree(d_Vt);
        cudaFree(d_Q); cudaFree(d_K_gain); cudaFree(d_innov);
        cudaFree(d_theta); cudaFree(d_ht); cudaFree(d_Ht); cudaFree(d_L_Ht); cudaFree(d_z);
        cudaFree(d_F1t); cudaFree(d_F2t); cudaFree(d_y1); cudaFree(d_y2);
        cudaFree(d_residual); cudaFree(d_rsd_sq); cudaFree(d_state_rsd);
        cudaFree(d_tmp1); cudaFree(d_tmp2); cudaFree(d_tmp3); cudaFree(d_tmp4); cudaFree(d_tmp5);
        cudaFree(d_info); cudaFree(d_ipiv); cudaFree(d_work);
        for (auto p : d_m_flt_all) cudaFree(p);
        for (auto p : d_C_flt_all) cudaFree(p);
        for (auto p : d_m_prd_all) cudaFree(p);
        for (auto p : d_C_prd_all) cudaFree(p);
        cublasDestroy(cublas);
        cusolverDnDestroy(solver);
        curandDestroyGenerator(curand_gen);
    }

    void setGW(const MatrixXd& G, const MatrixXd& W) {
        GS_CUDA_CHECK(cudaMemcpy(d_G, G.data(), (size_t)n_state*n_state*sizeof(double), cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_W, W.data(), (size_t)n_state*n_state*sizeof(double), cudaMemcpyHostToDevice));
    }

    void initialize(const VectorXd& m0, const MatrixXd& C0) {
        GS_CUDA_CHECK(cudaMemcpy(d_m_flt_all[0], m0.data(), n_state*sizeof(double), cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_C_flt_all[0], C0.data(), (size_t)n_state*n_state*sizeof(double), cudaMemcpyHostToDevice));
    }

    void filterStep(int t, const MatrixXd& Ftt_h, const VectorXd& yt_h, const MatrixXd& Vt_h) {
        int n_obs = (int)yt_h.size();
        GS_CUDA_CHECK(cudaMemcpy(d_Ftt, Ftt_h.data(),       (size_t)n_obs*n_state*sizeof(double), cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_yt,  yt_h.data(),        n_obs*sizeof(double),                  cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_Vt,  Vt_h.data(),        (size_t)n_obs*n_obs*sizeof(double),   cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_m_flt, d_m_flt_all[t],   n_state*sizeof(double),               cudaMemcpyDeviceToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_C_flt, d_C_flt_all[t],   (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));

        double one=1.0, zero=0.0, neg1=-1.0;

        // PREDICTION: m_prd = G * m_flt
        GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_state, n_state,
                                    &one, d_G, n_state, d_m_flt, 1, &zero, d_m_prd, 1));
        // C_prd = G * C_flt * G^T + W
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state, n_state,
                                    &one, d_G, n_state, d_C_flt, n_state, &zero, d_tmp1, n_state));
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                    n_state, n_state, n_state,
                                    &one, d_tmp1, n_state, d_G, n_state, &zero, d_C_prd, n_state));
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state,
                                    &one, d_C_prd, n_state, &one, d_W, n_state, d_C_prd, n_state));

        GS_CUDA_CHECK(cudaMemcpy(d_m_prd_all[t], d_m_prd, n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_C_prd_all[t], d_C_prd, (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));

        // UPDATE: innovation = y - Ftt * m_prd
        GS_CUDA_CHECK(cudaMemcpy(d_innov, d_yt, n_obs*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_obs, n_state,
                                    &neg1, d_Ftt, n_obs, d_m_prd, 1, &one, d_innov, 1));
        // Q = Ftt * C_prd * Ftt^T + Vt
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_obs, n_state, n_state,
                                    &one, d_Ftt, n_obs, d_C_prd, n_state, &zero, d_tmp2, n_obs));
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                    n_obs, n_obs, n_state,
                                    &one, d_tmp2, n_obs, d_Ftt, n_obs, &zero, d_Q, n_obs));
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_obs, n_obs, &one, d_Q, n_obs, &one, d_Vt, n_obs, d_Q, n_obs));
        int blk=(n_obs+255)/256;
        gs_add_diag_reg<<<blk,256>>>(d_Q, n_obs, 1e-8);
        GS_CUDA_CHECK(cudaDeviceSynchronize());

        // Kalman gain K = (C_prd * Ftt^T) * Q^-1  →  solve Q * K^T = (Ftt * C_prd^T)^T
        GS_CUDA_CHECK(cudaMemcpy(d_tmp3, d_Q, (size_t)n_obs*n_obs*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUSOLVER_CHECK(cusolverDnDgetrf(solver, n_obs, n_obs, d_tmp3, n_obs, d_work, d_ipiv, d_info));
        GS_CUDA_CHECK(cudaMemcpy(d_K_gain, d_tmp2, (size_t)n_obs*n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUSOLVER_CHECK(cusolverDnDgetrs(solver, CUBLAS_OP_N, n_obs, n_state,
                                           d_tmp3, n_obs, d_ipiv, d_K_gain, n_obs, d_info));
        // Transpose K from (n_obs × n_state) to (n_state × n_obs)
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                    n_state, n_obs, &one, d_K_gain, n_obs, &zero, d_tmp4, n_state, d_tmp4, n_state));
        GS_CUDA_CHECK(cudaMemcpy(d_K_gain, d_tmp4, (size_t)n_state*n_obs*sizeof(double), cudaMemcpyDeviceToDevice));

        // m_flt = m_prd + K * innov
        GS_CUDA_CHECK(cudaMemcpy(d_m_flt, d_m_prd, n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_state, n_obs,
                                    &one, d_K_gain, n_state, d_innov, 1, &one, d_m_flt, 1));
        // C_flt = C_prd - K * Ftt * C_prd
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state, n_obs,
                                    &one, d_K_gain, n_state, d_Ftt, n_obs, &zero, d_tmp1, n_state));
        GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state, n_state,
                                    &one, d_tmp1, n_state, d_C_prd, n_state, &zero, d_tmp3, n_state));
        GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n_state, n_state,
                                    &one, d_C_prd, n_state, &neg1, d_tmp3, n_state, d_C_flt, n_state));

        GS_CUDA_CHECK(cudaMemcpy(d_m_flt_all[t+1], d_m_flt, n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUDA_CHECK(cudaMemcpy(d_C_flt_all[t+1], d_C_flt, (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));
    }

    void backwardSample(int T) {
        double one=1.0, zero=0.0, neg1=-1.0;

        // Sample theta[T-1] ~ N(m_flt[T], C_flt[T])
        GS_CUDA_CHECK(cudaMemcpy(d_Ht, d_C_flt_all[T], (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, n_state, d_Ht, n_state, d_work, lwork, d_info));
        GS_CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_z, n_state, 0.0, 1.0));
        GS_CUDA_CHECK(cudaMemcpy(d_ht, d_m_flt_all[T], n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        GS_CUBLAS_CHECK(cublasDtrmv(cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                    n_state, d_Ht, n_state, d_z, 1));
        GS_CUBLAS_CHECK(cublasDaxpy(cublas, n_state, &one, d_z, 1, d_ht, 1));
        GS_CUDA_CHECK(cudaMemcpy(d_theta+(T-1)*n_state, d_ht, n_state*sizeof(double), cudaMemcpyDeviceToDevice));

        for (int t = T-2; t >= 0; t--) {
            GS_CUDA_CHECK(cudaMemcpy(d_m_flt, d_m_flt_all[t+1], n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUDA_CHECK(cudaMemcpy(d_C_flt, d_C_flt_all[t+1], (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUDA_CHECK(cudaMemcpy(d_m_prd, d_m_prd_all[t+1], n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUDA_CHECK(cudaMemcpy(d_C_prd, d_C_prd_all[t+1], (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));

            // diff = theta[t+1] - m_prd[t+1]
            GS_CUDA_CHECK(cudaMemcpy(d_innov, d_theta+(t+1)*n_state, n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUBLAS_CHECK(cublasDaxpy(cublas, n_state, &neg1, d_m_prd, 1, d_innov, 1));

            // C_prd^-1 * diff
            GS_CUDA_CHECK(cudaMemcpy(d_tmp3, d_C_prd, (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUSOLVER_CHECK(cusolverDnDgetrf(solver, n_state, n_state, d_tmp3, n_state, d_work, d_ipiv, d_info));
            GS_CUDA_CHECK(cudaMemcpy(d_tmp1, d_innov, n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUSOLVER_CHECK(cusolverDnDgetrs(solver, CUBLAS_OP_N, n_state, 1, d_tmp3, n_state, d_ipiv, d_tmp1, n_state, d_info));

            // G^T * (C_prd^-1 * diff)
            GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_T, n_state, n_state,
                                        &one, d_G, n_state, d_tmp1, 1, &zero, d_tmp2, 1));
            // ht = m_flt + C_flt * G^T * C_prd^-1 * diff
            GS_CUDA_CHECK(cudaMemcpy(d_ht, d_m_flt, n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_state, n_state,
                                        &one, d_C_flt, n_state, d_tmp2, 1, &one, d_ht, 1));

            // Ht = C_flt - C_flt * G^T * C_prd^-1 * G * C_flt
            GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                        n_state, n_state, n_state,
                                        &one, d_G, n_state, d_C_flt, n_state, &zero, d_tmp4, n_state));
            GS_CUDA_CHECK(cudaMemcpy(d_tmp3, d_C_prd, (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUSOLVER_CHECK(cusolverDnDgetrf(solver, n_state, n_state, d_tmp3, n_state, d_work, d_ipiv, d_info));
            GS_CUDA_CHECK(cudaMemcpy(d_tmp5, d_tmp4, (size_t)n_state*n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUSOLVER_CHECK(cusolverDnDgetrs(solver, CUBLAS_OP_N, n_state, n_state, d_tmp3, n_state, d_ipiv, d_tmp5, n_state, d_info));
            GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                        n_state, n_state, n_state,
                                        &one, d_C_flt, n_state, d_G, n_state, &zero, d_tmp4, n_state));
            GS_CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                        n_state, n_state, n_state,
                                        &one, d_tmp4, n_state, d_tmp5, n_state, &zero, d_tmp3, n_state));
            GS_CUBLAS_CHECK(cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                        n_state, n_state,
                                        &one, d_C_flt, n_state, &neg1, d_tmp3, n_state, d_Ht, n_state));

            GS_CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, n_state, d_Ht, n_state, d_work, lwork, d_info));
            GS_CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_z, n_state, 0.0, 1.0));
            GS_CUBLAS_CHECK(cublasDtrmv(cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                        n_state, d_Ht, n_state, d_z, 1));
            GS_CUBLAS_CHECK(cublasDaxpy(cublas, n_state, &one, d_z, 1, d_ht, 1));
            GS_CUDA_CHECK(cudaMemcpy(d_theta+t*n_state, d_ht, n_state*sizeof(double), cudaMemcpyDeviceToDevice));
        }
    }

    void computeResiduals(int T,
                          const vector<VectorXd>& y1, const vector<VectorXd>& y2,
                          const vector<MatrixXd>& F1t, const vector<MatrixXd>& F2t,
                          double& y1_ss, double& y1_n,
                          double& y2_ss, double& y2_n,
                          MatrixXd& state_rsd)
    {
        double one=1.0, neg1=-1.0;
        y1_ss = y1_n = y2_ss = y2_n = 0.0;
        int half = n_state / 2;

        for (int t = 0; t < T; t++) {
            int n1 = (int)y1[t].size(), n2 = (int)y2[t].size();
            auto residual_sq = [&](double* d_y_h, double* d_F_h,
                                   const VectorXd& yh, const MatrixXd& Fh,
                                   int n, double& ss, double& cnt) {
                if (n == 0) return;
                GS_CUDA_CHECK(cudaMemcpy(d_F_h, Fh.data(), (size_t)n*half*sizeof(double), cudaMemcpyHostToDevice));
                GS_CUDA_CHECK(cudaMemcpy(d_y_h, yh.data(), n*sizeof(double), cudaMemcpyHostToDevice));
                GS_CUDA_CHECK(cudaMemcpy(d_residual, d_y_h, n*sizeof(double), cudaMemcpyDeviceToDevice));
                GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n, half,
                                            &neg1, d_F_h, n, d_theta+t*n_state, 1, &one, d_residual, 1));
                GS_CUDA_CHECK(cudaMemset(d_rsd_sq, 0, sizeof(double)));
                int blk = (n+255)/256;
                gs_squared_norm<<<blk,256>>>(d_residual, d_rsd_sq, n);
                GS_CUDA_CHECK(cudaDeviceSynchronize());
                double v; GS_CUDA_CHECK(cudaMemcpy(&v, d_rsd_sq, sizeof(double), cudaMemcpyDeviceToHost));
                ss += v; cnt += n;
            };
            residual_sq(d_y1, d_F1t, y1[t], F1t[t], n1, y1_ss, y1_n);
            residual_sq(d_y2, d_F2t, y2[t], F2t[t], n2, y2_ss, y2_n);
        }

        double zero=0.0, one2=1.0;
        state_rsd.resize(n_state, T > 1 ? T-1 : 1);
        vector<double> buf((size_t)n_state*(T > 1 ? T-1 : 1));
        for (int t = 0; t < T-1; t++) {
            GS_CUDA_CHECK(cudaMemcpy(d_state_rsd+t*n_state, d_theta+(t+1)*n_state,
                                     n_state*sizeof(double), cudaMemcpyDeviceToDevice));
            GS_CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n_state, n_state,
                                        &neg1, d_G, n_state, d_theta+t*n_state, 1,
                                        &one2, d_state_rsd+t*n_state, 1));
        }
        GS_CUDA_CHECK(cudaMemcpy(buf.data(), d_state_rsd, buf.size()*sizeof(double), cudaMemcpyDeviceToHost));
        for (int t = 0; t < T-1; t++)
            for (int i = 0; i < n_state; i++)
                state_rsd(i,t) = buf[t*n_state+i];
    }

    // Extract filtered means m_flt[1..T] → out[0..T-1]
    void getMflt(int T, vector<VectorXd>& out) {
        out.resize(T);
        vector<double> buf(n_state);
        for (int t = 0; t < T; t++) {
            GS_CUDA_CHECK(cudaMemcpy(buf.data(), d_m_flt_all[t+1],
                                     n_state*sizeof(double), cudaMemcpyDeviceToHost));
            out[t] = Eigen::Map<VectorXd>(buf.data(), n_state);
        }
    }
};

// ---------------------------------------------------------------------------
// run_gibbs_pipeline
// ---------------------------------------------------------------------------
void run_gibbs_pipeline(
    int T, int N_sample, int seed,
    const MatrixXd& expG,
    const VectorXd& m0,
    const MatrixXd& C0,
    const vector<MatrixXd>& Ftt_all,
    const vector<VectorXd>& yt_all,
    const vector<MatrixXd>& Vt_all,
    const vector<int>& n1_per_t,
    int max_obs,
    vector<VectorXd>& m_flt,
    vector<VectorXd>& m_flt_gibbs)
{
    cout << "=== Gibbs FFBS2 Pipeline ===" << endl;
    cout << "T=" << T << "  N_sample=" << N_sample << "  n_state=" << m0.size() << endl;

    mt19937 gen(seed);
    int n_state = (int)m0.size();      // 2*N^2

    // Initial W, sigma parameters
    // sigma: fixed at prior mode (0.05) rather than sampling degenerate InvGamma(1,0.1)
    // W: initialized to Wishart mean = nu*Phi = nu*0.01*I, matching R's W[[1]] <- rwish(nu,Phi)
    //    which has mean nu*Phi = 8*I.  The GPU previously used riwish(800, 0.01*I) which is
    //    near-singular (nu==p) and caused a biased first filter pass.
    const double alpha1_init=1.0, alpha2_init=1.0;
    const double beta1_init=0.1,  beta2_init=0.1;
    const int nu_init = C0.rows();
    MatrixXd Phi_init = 0.01 * MatrixXd::Identity(C0.rows(), C0.cols());

    double sigma1 = beta1_init / (alpha1_init + 1.0);  // prior mode of InvGamma(1,0.1) = 0.05
    double sigma2 = beta2_init / (alpha2_init + 1.0);
    // W[[1]] <- rwish(nu, Phi) matching R's Gibbs_FFBS2 initialization exactly.
    // The random Wishart sample has non-zero off-diagonal blocks between alpha1 and
    // alpha2 sub-blocks, which creates coupling from the very first filter pass —
    // critical for alpha2 (bias field) to develop spatial structure.
    cout << "  Sampling initial W ~ Wishart(" << nu_init << ", Phi)..." << flush;
    MatrixXd W = rwish_cpu(gen, nu_init, Phi_init);
    cout << " done." << endl;

    double alpha1=alpha1_init, alpha2=alpha2_init;
    double beta1=beta1_init,   beta2=beta2_init;
    int nu = nu_init;
    MatrixXd Phi = Phi_init;

    // Half-state dimension (number of Fourier basis columns N^2)
    int half_state = n_state / 2;

    // Split Ftt into F1t, F2t using n1_per_t for exact boundary
    vector<VectorXd> y1(T), y2(T);
    vector<MatrixXd> F1t(T), F2t(T);
    for (int t = 0; t < T; t++) {
        int n1 = n1_per_t[t];
        int n2 = (int)yt_all[t].size() - n1;
        y1[t] = yt_all[t].head(n1);
        y2[t] = yt_all[t].tail(n2);
        F1t[t] = Ftt_all[t].topRows(n1).leftCols(half_state);
        F2t[t] = Ftt_all[t].bottomRows(n2).leftCols(half_state);
    }

    GibbsSamplerPipeline sampler(n_state, max_obs, T, (unsigned long long)seed);

    auto t_start = chrono::high_resolution_clock::now();

    for (int k = 0; k < N_sample; k++) {
        if (k % max(1, N_sample/10) == 0)
            cout << "  Iteration " << k+1 << "/" << N_sample
                 << "  sigma1=" << sigma1 << "  sigma2=" << sigma2 << endl;

        sampler.setGW(expG, W);
        sampler.initialize(m0, C0);

        // Forward filter
        for (int t = 0; t < T; t++) {
            int n_obs = (int)yt_all[t].size();
            int n1    = n1_per_t[t];
            int n2    = n_obs - n1;
            VectorXd sigma2t(n_obs);
            sigma2t.head(n1).setConstant(sigma1);
            sigma2t.tail(n2).setConstant(sigma2);
            sampler.filterStep(t, Ftt_all[t], yt_all[t], sigma2t.asDiagonal());
        }

        // Backward sample
        sampler.backwardSample(T);

        // Residuals for parameter update
        double y1_ss, y1_n, y2_ss, y2_n;
        MatrixXd srsd;
        sampler.computeResiduals(T, y1, y2, F1t, F2t, y1_ss, y1_n, y2_ss, y2_n, srsd);

        alpha1 += y1_n/2.0;  beta1 += y1_ss/2.0;
        alpha2 += y2_n/2.0;  beta2 += y2_ss/2.0;
        nu += (T-1);
        Phi += srsd * srsd.transpose();

        sigma1 = rinvgamma(gen, alpha1, beta1);
        sigma2 = rinvgamma(gen, alpha2, beta2);
        W = riwish(gen, nu, Phi);
    }

    auto t_end = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(t_end - t_start).count();
    cout << "Gibbs done: " << secs << " s  (" << secs/N_sample << " s/iter)" << endl;
    cout << "  Final params: sigma1=" << sigma1 << "  sigma2=" << sigma2 << endl;

    // Capture last Gibbs iteration's forward-filtered means.
    // These were produced with W built up through backward-sample cross-products,
    // so the alpha2 (bias) block retains spatial structure unlike the extra pass.
    sampler.getMflt(T, m_flt_gibbs);

    // Extra forward pass using the fully-converged final sigma1, sigma2, W.
    // The Gibbs loop's last iteration filtered with second-to-last parameter values;
    // this pass uses the final (best) estimates.
    sampler.setGW(expG, W);
    sampler.initialize(m0, C0);
    for (int t = 0; t < T; t++) {
        int n_obs = (int)yt_all[t].size();
        int n1    = n1_per_t[t];
        int n2    = n_obs - n1;
        VectorXd sigma2t(n_obs);
        sigma2t.head(n1).setConstant(sigma1);
        sigma2t.tail(n2).setConstant(sigma2);
        sampler.filterStep(t, Ftt_all[t], yt_all[t], sigma2t.asDiagonal());
    }

    // Extract m.flt from final forward pass
    sampler.getMflt(T, m_flt);
}
