#pragma once
#include <vector>

// GPU observation assembly — exact replica of util.R::OBS_ccl2() per time step.
//
// R call: OBS_ccl2(y1, y2, F)  where for each time step i:
//   tempt   = M_id(y[[i]])          → find non-NaN indices in y
//   Ft[[i]] = M %*% F               → select rows of F at observed locations
//   yc[[i]] = y[[i]][ms.id]         → extract non-NaN observations
//   id[[i]] = ms.id                 → 1-indexed location indices
//
// GPU equivalent (per time step, 0-indexed):
//   d_id[0..n_obs-1]             — 0-indexed locations of non-NaN values
//   d_yc[0..n_obs-1]             — the non-NaN observation values
//   d_Ft[0..n_obs-1][0..Ngrid-1] — rows of F at those locations
//
// d_F    : device, Nr² × Ngrid row-major (unchanged throughout)
// d_y    : device, Nr² doubles (NaN = missing) for this time step
// Nr2    : Nr*Nr (= 3600)
// Ngrid  : Fourier basis dimension (= 400)
// d_Ft   : device output, pre-allocated ≥ Nr2*Ngrid doubles
// d_yc   : device output, pre-allocated ≥ Nr2 doubles
// d_id   : device output, pre-allocated ≥ Nr2 ints (0-indexed)
// n_obs  : host output — number of non-NaN observations found
void obs_assembly_gpu(const double* d_F, const double* d_y,
                      int Nr2, int Ngrid,
                      double* d_Ft, double* d_yc, int* d_id,
                      int& n_obs);
