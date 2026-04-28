#pragma once
// GPU optical flow — exact port of R's SpatialVx::OF()
// Processes one (img1, img2) pair per call.
//
// h_img1, h_img2 : host, Nr*Nr row-major (row = lat, col = lon), may contain NaN
// h_speed, h_angle: host output, Nr*Nr, NaN at boundaries (rows/cols 0,1,Nr-2,Nr-1)
void optical_flow_gpu(const double* h_img1,
                      const double* h_img2,
                      int Nr, int W,
                      double* h_speed,
                      double* h_angle);
