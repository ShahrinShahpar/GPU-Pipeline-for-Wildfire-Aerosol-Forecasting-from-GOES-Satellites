// optical_flow.cu — GPU optical flow (SpatialVx::OF replica), pipeline version.
// Stripped from Physics-Informed.../optical_flow_r_complete.cu (main removed).
// Adds optical_flow_gpu() host wrapper callable from main_pipeline.cu.
//
// Build as standalone verifier:
//   nvcc -arch=sm_70 -std=c++14 -O3 -I. -DOF_VERIFY_MAIN \
//        optical_flow.cu csv_io.cu -o of_verify
//
// Usage:
//   ./of_verify initial.csv final.csv out_speed.csv out_angle.csv ref_speed.csv ref_angle.csv

#include "optical_flow.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define OF_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error [optical_flow] %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// ============================================================================
// Device helpers — unchanged from optical_flow_r_complete.cu
// ============================================================================

__device__ void compute_window_grads1(
    const double* img, int center_row, int center_col, int W, int Nr,
    double* Ir_win, double* Ic_win,
    double* Irr_win, double* Icc_win, double* Irc_win,
    int* n_valid)
{
    int half_W = W / 2, count = 0;
    for (int wy = -half_W; wy <= half_W; wy++) {
        for (int wx = -half_W; wx <= half_W; wx++) {
            int row = center_row + wy, col = center_col + wx;
            if (row >= 0 && row < Nr-2 && col >= 0 && col < Nr-2) {
                int idx = row * Nr + col;
                Ir_win[count]  = img[(row+1)*Nr+col] - img[idx];
                Ic_win[count]  = img[row*Nr+(col+1)] - img[idx];
                Irr_win[count] = img[(row+2)*Nr+col] - 2.0*img[(row+1)*Nr+col] + img[idx];
                Icc_win[count] = img[row*Nr+(col+2)] - 2.0*img[row*Nr+(col+1)] + img[idx];
                Irc_win[count] = img[(row+1)*Nr+(col+1)] - img[row*Nr+(col+1)]
                               - img[(row+1)*Nr+col] + img[idx];
                count++;
            }
        }
    }
    *n_valid = count;
}

__device__ void compute_linear_initial_guess(
    const double* initial_win, const double* final_win,
    const double* Ir_win, const double* Ic_win,
    int n_points, double* a0, double* a1, double* a2)
{
    double sum_Ir=0, sum_Ic=0, sum_y=0;
    double sum_Ir2=0, sum_Ic2=0, sum_IrIc=0, sum_yIr=0, sum_yIc=0;
    for (int i = 0; i < n_points; i++) {
        double y = final_win[i] - initial_win[i];
        double x2 = Ir_win[i], x3 = Ic_win[i];
        sum_y += y; sum_Ir += x2; sum_Ic += x3;
        sum_Ir2 += x2*x2; sum_Ic2 += x3*x3; sum_IrIc += x2*x3;
        sum_yIr += y*x2; sum_yIc += y*x3;
    }
    double n = (double)n_points;
    double A[3][4];
    A[0][0]=n;       A[0][1]=sum_Ir;   A[0][2]=sum_Ic;   A[0][3]=sum_y;
    A[1][0]=sum_Ir;  A[1][1]=sum_Ir2;  A[1][2]=sum_IrIc; A[1][3]=sum_yIr;
    A[2][0]=sum_Ic;  A[2][1]=sum_IrIc; A[2][2]=sum_Ic2;  A[2][3]=sum_yIc;
    for (int k = 0; k < 3; k++) {
        int max_row = k; double max_val = fabs(A[k][k]);
        for (int i = k+1; i < 3; i++)
            if (fabs(A[i][k]) > max_val) { max_val = fabs(A[i][k]); max_row = i; }
        if (max_row != k)
            for (int j = 0; j < 4; j++) { double tmp=A[k][j]; A[k][j]=A[max_row][j]; A[max_row][j]=tmp; }
        if (fabs(A[k][k]) < 1e-12) { *a0=0; *a1=0; *a2=0; return; }
        for (int i = k+1; i < 3; i++) {
            double f = A[i][k] / A[k][k];
            for (int j = k; j < 4; j++) A[i][j] -= f * A[k][j];
        }
    }
    *a2 = A[2][3] / A[2][2];
    *a1 = (A[1][3] - A[1][2]*(*a2)) / A[1][1];
    *a0 = (A[0][3] - A[0][1]*(*a1) - A[0][2]*(*a2)) / A[0][0];
}

// ============================================================================
// Main optical flow kernel
// ============================================================================
__global__ void optical_flow_r_kernel(
    const double* img1, const double* img2,
    double* speed, double* angle,
    int Nr, int W,
    double mean_field,
    double mean_Ir_global, double mean_Ic_global,
    double mean_Irr_global, double mean_Icc_global, double mean_Irc_global)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Nr || col >= Nr) return;
    int idx = row * Nr + col;
    if (row < 1 || row > Nr-3 || col < 1 || col > Nr-3) {
        speed[idx] = NAN; angle[idx] = NAN; return;
    }

    const int MAX_WIN = 15*15;
    double initial_win[MAX_WIN], final_win[MAX_WIN];
    double Ir_win[MAX_WIN], Ic_win[MAX_WIN];
    double Irr_win[MAX_WIN], Icc_win[MAX_WIN], Irc_win[MAX_WIN];

    int half_W = W / 2;
    int row_start = (row-half_W < 0) ? 0 : row-half_W;
    int row_end   = (row+half_W >= Nr) ? Nr-1 : row+half_W;
    int col_start = (col-half_W < 0) ? 0 : col-half_W;
    int col_end   = (col+half_W >= Nr) ? Nr-1 : col+half_W;
    int crop_h = row_end - row_start + 1;
    int crop_w = col_end - col_start + 1;

    double crop_img1[MAX_WIN], crop_img2[MAX_WIN];
    int n_crop = 0;
    for (int r = row_start; r <= row_end; r++)
        for (int c = col_start; c <= col_end; c++) {
            crop_img1[n_crop] = img1[r*Nr+c];
            crop_img2[n_crop] = img2[r*Nr+c];
            n_crop++;
        }

    int n_valid = 0;
    for (int r = 0; r < crop_h; r++) {
        for (int c = 0; c < crop_w; c++) {
            if (r+2 >= crop_h || c+2 >= crop_w) continue;
            int ci = r*crop_w + c;
            initial_win[n_valid] = crop_img1[ci];
            final_win[n_valid]   = crop_img2[ci];
            Ir_win[n_valid]  = crop_img1[(r+1)*crop_w+c] - crop_img1[ci];
            Ic_win[n_valid]  = crop_img1[r*crop_w+(c+1)] - crop_img1[ci];
            Irr_win[n_valid] = crop_img1[(r+2)*crop_w+c] - 2.0*crop_img1[(r+1)*crop_w+c] + crop_img1[ci];
            Icc_win[n_valid] = crop_img1[r*crop_w+(c+2)] - 2.0*crop_img1[r*crop_w+(c+1)] + crop_img1[ci];
            Irc_win[n_valid] = crop_img1[(r+1)*crop_w+(c+1)] - crop_img1[r*crop_w+(c+1)]
                             - crop_img1[(r+1)*crop_w+c] + crop_img1[ci];
            n_valid++;
        }
    }

    if (n_valid < 10) { speed[idx] = NAN; angle[idx] = NAN; return; }

    for (int i = 0; i < n_valid; i++) {
        initial_win[i] -= mean_field;   final_win[i] -= mean_field;
        Ir_win[i] -= mean_Ir_global;    Ic_win[i] -= mean_Ic_global;
        Irr_win[i] -= mean_Irr_global;  Icc_win[i] -= mean_Icc_global;
        Irc_win[i] -= mean_Irc_global;
    }

    double a0, a1, a2;
    compute_linear_initial_guess(initial_win, final_win, Ir_win, Ic_win,
                                 n_valid, &a0, &a1, &a2);
    double vr = -a1, vc = -a2;
    speed[idx] = sqrt(vr*vr + vc*vc);

    double ang_deg;
    if      (vc > 0 && vr >= 0) ang_deg = atan(vr/vc)*180.0/M_PI;
    else if (vc < 0 && vr >= 0) ang_deg = 180.0 + atan(vr/vc)*180.0/M_PI;
    else if (vc < 0 && vr <= 0) ang_deg = 180.0 + atan(vr/vc)*180.0/M_PI;
    else if (vc > 0 && vr <= 0) ang_deg = 360.0 + atan(vr/vc)*180.0/M_PI;
    else if (vc == 0 && vr > 0) ang_deg = 90.0;
    else if (vc == 0 && vr < 0) ang_deg = 270.0;
    else                         ang_deg = 0.0;
    angle[idx] = ang_deg;
}

// ============================================================================
// Host wrapper — called from main_pipeline.cu
// ============================================================================
void optical_flow_gpu(const double* h_img1, const double* h_img2,
                      int Nr, int W,
                      double* h_speed, double* h_angle)
{
    size_t img_bytes = (size_t)Nr * Nr * sizeof(double);
    double *d_img1, *d_img2, *d_speed, *d_angle;
    OF_CHECK(cudaMalloc(&d_img1,  img_bytes));
    OF_CHECK(cudaMalloc(&d_img2,  img_bytes));
    OF_CHECK(cudaMalloc(&d_speed, img_bytes));
    OF_CHECK(cudaMalloc(&d_angle, img_bytes));
    OF_CHECK(cudaMemcpy(d_img1, h_img1, img_bytes, cudaMemcpyHostToDevice));
    OF_CHECK(cudaMemcpy(d_img2, h_img2, img_bytes, cudaMemcpyHostToDevice));

    // CPU: compute mean field of img2 and global mean gradients (matching R)
    double mean_field = 0.0;
    int cnt_v = 0;
    for (int i = 0; i < Nr*Nr; i++)
        if (!isnan(h_img2[i])) { mean_field += h_img2[i]; cnt_v++; }
    if (cnt_v > 0) mean_field /= cnt_v;

    double sum_Ir=0, sum_Ic=0, sum_Irr=0, sum_Icc=0, sum_Irc=0;
    int cnt_g = 0;
    for (int r = 0; r < Nr-2; r++) {
        for (int c = 0; c < Nr-2; c++) {
            int idx = r*Nr+c;
            double Ir  = h_img2[(r+1)*Nr+c] - h_img2[idx];
            double Ic  = h_img2[r*Nr+(c+1)] - h_img2[idx];
            double Irr = h_img2[(r+2)*Nr+c] - 2*h_img2[(r+1)*Nr+c] + h_img2[idx];
            double Icc = h_img2[r*Nr+(c+2)] - 2*h_img2[r*Nr+(c+1)] + h_img2[idx];
            double Irc = h_img2[(r+1)*Nr+(c+1)] - h_img2[r*Nr+(c+1)]
                       - h_img2[(r+1)*Nr+c] + h_img2[idx];
            if (!isnan(Ir)) { sum_Ir+=Ir; sum_Ic+=Ic; sum_Irr+=Irr; sum_Icc+=Icc; sum_Irc+=Irc; cnt_g++; }
        }
    }
    double mIr  = cnt_g > 0 ? sum_Ir/cnt_g  : 0;
    double mIc  = cnt_g > 0 ? sum_Ic/cnt_g  : 0;
    double mIrr = cnt_g > 0 ? sum_Irr/cnt_g : 0;
    double mIcc = cnt_g > 0 ? sum_Icc/cnt_g : 0;
    double mIrc = cnt_g > 0 ? sum_Irc/cnt_g : 0;

    dim3 block(16, 16), grid((Nr+15)/16, (Nr+15)/16);
    optical_flow_r_kernel<<<grid, block>>>(
        d_img1, d_img2, d_speed, d_angle,
        Nr, W, mean_field, mIr, mIc, mIrr, mIcc, mIrc);
    OF_CHECK(cudaDeviceSynchronize());

    OF_CHECK(cudaMemcpy(h_speed, d_speed, img_bytes, cudaMemcpyDeviceToHost));
    OF_CHECK(cudaMemcpy(h_angle, d_angle, img_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_img1); cudaFree(d_img2);
    cudaFree(d_speed); cudaFree(d_angle);
}

// ============================================================================
// Standalone verifier — compiled only when -DOF_VERIFY_MAIN is set
// ============================================================================
#ifdef OF_VERIFY_MAIN

#include "csv_io.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>

static void write_csv_17(const std::string& path, const double* data, int Nr)
{
    std::ofstream f(path);
    f << std::setprecision(17);
    for (int j = 0; j < Nr; j++) { if (j > 0) f << ','; f << 'V' << (j+1); }
    f << '\n';
    for (int i = 0; i < Nr; i++) {
        for (int j = 0; j < Nr; j++) {
            if (j > 0) f << ',';
            double v = data[i * Nr + j];
            if (std::isnan(v)) f << "NaN"; else f << v;
        }
        f << '\n';
    }
}

static void compare_field(const std::vector<double>& gpu,
                           const std::vector<double>& ref,
                           int Nr, const char* name)
{
    int total = Nr * Nr, nan_mm = 0, n_cmp = 0, n_fail = 0;
    double max_d = 0.0, sum_d = 0.0;
    for (int i = 0; i < total; i++) {
        bool gn = std::isnan(gpu[i]), rn = std::isnan(ref[i]);
        if (gn || rn) { if (gn != rn) ++nan_mm; continue; }
        double d = std::fabs(gpu[i] - ref[i]);
        max_d = std::max(max_d, d);
        sum_d += d; ++n_cmp;
        if (d > 1e-6) ++n_fail;
    }
    printf("  %s : NaN_mm=%d  max|diff|=%.3e  mean|diff|=%.3e  cells>1e-6=%d/%d  %s\n",
           name, nan_mm, max_d, n_cmp > 0 ? sum_d/n_cmp : 0.0,
           n_fail, n_cmp,
           (nan_mm == 0 && max_d < 1e-6) ? "PASS" : "FAIL");
}

int main(int argc, char* argv[])
{
    if (argc != 7) {
        fprintf(stderr,
            "Usage: %s initial.csv final.csv out_speed.csv out_angle.csv ref_speed.csv ref_angle.csv\n"
            "\n"
            "  initial.csv  — 60×60 KNN-imputed image (R write.csv format, row=lon col=lat)\n"
            "  final.csv    — 60×60 KNN-imputed image (next time step)\n"
            "  out_speed.csv — GPU OF speed output (written here)\n"
            "  out_angle.csv — GPU OF angle output in degrees (written here)\n"
            "  ref_speed.csv — R SpatialVx::OF speed reference\n"
            "  ref_angle.csv — R SpatialVx::OF angle reference (degrees)\n",
            argv[0]);
        return 1;
    }

    const int Nr = 60, W = 15;
    int nr = 0, nc = 0;

    std::vector<double> h_init, h_final;
    try {
        h_init  = read_csv_flat(argv[1], nr, nc, true);
        h_final = read_csv_flat(argv[2], nr, nc, true);
    } catch (const std::exception& e) {
        fprintf(stderr, "Read error: %s\n", e.what()); return 1;
    }
    if (nr != Nr || nc != Nr) {
        fprintf(stderr, "Expected %dx%d, got %dx%d\n", Nr, Nr, nr, nc); return 1;
    }
    printf("Input  : %s / %s  (%dx%d)\n", argv[1], argv[2], nr, nc);

    std::vector<double> h_speed(Nr * Nr), h_angle(Nr * Nr);
    optical_flow_gpu(h_init.data(), h_final.data(), Nr, W,
                     h_speed.data(), h_angle.data());

    write_csv_17(argv[3], h_speed.data(), Nr);
    write_csv_17(argv[4], h_angle.data(), Nr);
    printf("Written: %s  %s\n", argv[3], argv[4]);

    // Compare vs reference
    std::vector<double> ref_speed, ref_angle;
    try {
        ref_speed = read_csv_flat(argv[5], nr, nc, true);
        ref_angle = read_csv_flat(argv[6], nr, nc, true);
    } catch (const std::exception& e) {
        fprintf(stderr, "Ref read error: %s\n", e.what()); return 1;
    }
    printf("\n=== Optical Flow Verification (one pair) ===\n");
    compare_field(h_speed, ref_speed, Nr, "speed");
    compare_field(h_angle, ref_angle, Nr, "angle");
    return 0;
}

#endif // OF_VERIFY_MAIN
