#pragma once

// GPU KNN imputation — exact replica of bnstruct::knn.impute(k=10)
//
// Algorithm (mirrors R call in application.R):
//   knn.impute(tempt, k=10, cat.var=1:ncol(tempt),
//              to.impute=1:nrow(tempt), using=1:nrow(tempt))
//
//   All columns are treated as CATEGORICAL → HEOM distance:
//     cat_dist(a, b) = 0  if a == b (exact float equality, both non-NaN)
//                    = 1  otherwise (values differ, or either is NaN)
//     d(r, s) = sqrt( sum_c cat_dist(r[c], s[c])^2 )
//
//   For each NaN column c in row r:
//     - Walk rows sorted by d(r, .) ascending
//     - Collect the first k rows that have non-NaN at column c
//     - Impute with mode of those k values
//     - For continuous floats (all distinct), mode = nearest neighbor's value
//
// d_aod : device pointer, Nr*Nr doubles (NaN for missing), stored row-major
//         d_aod[row * Nr + col]
// Nr    : grid side length (60)
// k     : number of nearest neighbors (10)
//
// Constraint: Nr <= 64.
void knn_impute_gpu(double* d_aod, int Nr, int k);
