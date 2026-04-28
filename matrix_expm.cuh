#pragma once

// GPU matrix exponential — Padé approximation degree 13 with scaling-and-squaring.
//
// Matches R's expm::expm(G) (default method = "Higham08").
//
// Algorithm: Higham 2008, "Functions of Matrices: Theory and Computation",
//            Algorithm 10.20 (Padé degree-13).
//
// d_A   : device pointer, N×N doubles, row-major input (modified in-place)
// d_out : device pointer, N×N doubles, row-major output  exp(A)
// N     : matrix dimension (= 400 for this pipeline)
void matrix_expm_gpu(const double* d_A, double* d_out, int N);
