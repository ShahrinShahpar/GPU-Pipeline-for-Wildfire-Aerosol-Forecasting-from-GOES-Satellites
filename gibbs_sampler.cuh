#pragma once
// GPU Gibbs FFBS2 sampler — pipeline interface.
// Adapted from Physics-Informed.../gibbs_step6.cu.
// Adds getMflt() for extracting filtered state means (m.flt in R).

#include <Eigen/Dense>
#include <vector>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Run Gibbs FFBS2 with T time steps.
//
// expG       : 2*N^2 x 2*N^2 augmented transition matrix
// m0         : 2*N^2-dim initial state mean  (rep(0.1, 2*N^2))
// C0         : 2*N^2 x 2*N^2 initial covariance  (0.01 * I)
// Ftt_all[t] : n_obs_t x 2*N^2 combined observation matrix for time t
//              (top n1_per_t[t] rows = F1t, bottom n2 rows = F2t; right half zero)
// yt_all[t]  : n_obs_t observations  (y1c then y2c stacked)
// Vt_all[t]  : n_obs_t x n_obs_t diagonal noise covariance  (sigma^2 * I)
// n1_per_t   : number of satellite-1 observations per time step
// max_obs    : max n_obs_t across all t  (for pre-allocation)
// N_sample   : number of Gibbs iterations
// seed       : random seed
//
// Returns two sets of filtered state means:
//   m_flt      : from the extra forward pass with final converged W/sigma
//                (best for alpha1/AOD reconstruction)
//   m_flt_gibbs: from the last Gibbs iteration's forward pass (W built up
//                off-diagonal coupling through backward samples)
//                (best for alpha2/bias field reconstruction)
void run_gibbs_pipeline(
    int T, int N_sample, int seed,
    const MatrixXd& expG,
    const VectorXd& m0,
    const MatrixXd& C0,
    const std::vector<MatrixXd>& Ftt_all,
    const std::vector<VectorXd>& yt_all,
    const std::vector<MatrixXd>& Vt_all,
    const std::vector<int>& n1_per_t,
    int max_obs,
    std::vector<VectorXd>& m_flt,        // output: from final extra forward pass
    std::vector<VectorXd>& m_flt_gibbs   // output: from last Gibbs iteration
);
