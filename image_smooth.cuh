#pragma once

// GPU implementation of fields::image.smooth — exact replica.
//
// R call in application.R / util.R:
//   image.smooth(mat)$z   (all defaults)
//
// Algorithm (mirrors fields::image.smooth + setup.image.smooth):
//   kernel.function = double.exp(d2) = 0.5 * exp(-d2)
//   where d2 = (xi^2 + yi^2) / aRange^2 is the scaled squared Euclidean dist.
//   Default: aRange=1, tol=1e-8.
//
//   Padded grid: M = N = 2*Nr (= 120 for Nr=60)
//   Kernel center at (M/2-1, N/2-1) = (59, 59) in 0-indexed.
//
//   W = FFT2(kernel) / FFT2(impulse_at_center) / (M*N)
//
//   Nadaraya-Watson normalization:
//     num = Re( IFFT2( FFT2(data_padded)  × W ) )[0:Nr, 0:Nr]
//     den = Re( IFFT2( FFT2(mask_padded)  × W ) )[0:Nr, 0:Nr]
//     result = num / den   (where den > tol, else NaN)
//
// d_in  : device pointer, Nr×Nr doubles (NaN for missing), row-major
// d_out : device pointer, Nr×Nr doubles (smoothed), row-major
// Nr    : grid side length (60)
// aRange: bandwidth (default 1.0 — matches R default)
// tol   : denominator threshold (default 1e-8 — matches R default)
void image_smooth_gpu(const double* d_in, double* d_out, int Nr,
                      double aRange = 1.0, double tol = 1e-8);
