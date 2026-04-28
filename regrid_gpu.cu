/*
 * regrid_gpu.cu  —  GPU-accelerated regridding of GOES-16 / GOES-17 AOD
 *
 * Part of gpuimplementation/ — pure GPU pipeline, zero R.
 *
 * Compiles two ways:
 *
 *   Callable module (default, for main_pipeline.cu):
 *     nvcc -O3 -arch=sm_70 -c regrid_gpu.cu -o regrid_gpu.o
 *
 *   Standalone verifier (reads .bin, writes .csv):
 *     nvcc -O3 -arch=sm_70 -DREGRID_STANDALONE_MAIN regrid_gpu.cu -o regrid_standalone
 *     ./regrid_standalone manifest_G16.txt output/G16
 *
 * ALGORITHM: exact replica of try/functions/regrid.R
 * -------------------------------------------------------
 * R does (for each of the 3600 output cells):
 *
 *   tempt <- dat %>% subset(
 *     lat  >= center_lat  - 0.5*res  &
 *     lat  <= center_lat  + 0.5*res  &
 *     long >= center_long - 0.5*res  &
 *     long <= center_long + 0.5*res )
 *   AOD[xyy] <- mean(tempt$AOD)
 *
 * Key subtleties replicated here:
 *  1. Inclusive bounds (>= AND <=) on BOTH sides of the cell  [R uses both]
 *  2. float64 (double) throughout so cell boundaries match R exactly
 *  3. Grid ordering: long varies fastest (expand.grid convention)
 *  4. Empty cells → NaN (R's mean() on empty vector = NaN)
 *
 * Binary format from preprocess.py:
 *   int32   n
 *   float64[n]  lats
 *   float64[n]  lons
 *   float64[n]  aods   (already clamped ≥ 0 and NaN-filtered)
 *
 * Output (standalone): csv_output/img_NNN.csv  with columns  long,lat,AOD
 */

#include "regrid_gpu.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

using steady_clock = std::chrono::steady_clock;
using duration_ms  = std::chrono::duration<double, std::milli>;
static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        steady_clock::now().time_since_epoch()).count();
}

#include <cuda_runtime.h>

/* -------------------------------------------------------------------------
 * Grid parameters — must match GOES16_data_clean_timed.R exactly
 * ---------------------------------------------------------------------- */
#define LON_MIN   (-124.0)
#define LON_MAX   (-121.6)
#define LAT_MIN   (  35.0)
#define LAT_MAX   (  37.4)
#define RESOLUTION (  0.04)

/* n.x = round((LON_MAX-LON_MIN)/RESOLUTION) = round(60) = 60 */
#define N_LON  60
/* n.y = round((LAT_MAX-LAT_MIN)/RESOLUTION) = round(60) = 60 */
#define N_LAT  60
#define N_GRID (N_LON * N_LAT)   /* 3600 */

/* -------------------------------------------------------------------------
 * Kernel: one thread per OUTPUT grid cell.
 *
 * Each thread scans ALL input points and accumulates those whose
 * lat/lon fall inside the cell's inclusive bounding box, exactly
 * as R does with >= and <=.
 *
 * All arithmetic is double-precision to match R's float64.
 *
 * Grid ordering: g = lat_i * N_LON + lon_i  (long varies fastest),
 * matching R's expand.grid(long=..., lat=...).
 * ---------------------------------------------------------------------- */
__global__ void k_regrid(
    const double* __restrict__ lats,
    const double* __restrict__ lons,
    const double* __restrict__ aods,
    int           n_points,
    double*       means)          /* output, one value per grid cell */
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N_GRID) return;

    int lat_i = g / N_LON;
    int lon_i = g % N_LON;

    /* Cell centre — exact same formula as R:
     *   pixel.center.x[k] = k*res - 0.5*res + lon_min   (k = 1..n.x, 1-indexed)
     * which in 0-indexed form is (lon_i + 0.5)*res + lon_min             */
    double cell_lat = (lat_i + 0.5) * RESOLUTION + LAT_MIN;
    double cell_lon = (lon_i + 0.5) * RESOLUTION + LON_MIN;
    double half     = 0.5 * RESOLUTION;

    /* Inclusive bounds, matching R's >= and <=  */
    double lat_lo = cell_lat - half;
    double lat_hi = cell_lat + half;
    double lon_lo = cell_lon - half;
    double lon_hi = cell_lon + half;

    double sum = 0.0;
    int    cnt = 0;

    for (int i = 0; i < n_points; ++i) {
        double lat = lats[i];
        double lon = lons[i];
        if (lat >= lat_lo && lat <= lat_hi &&
            lon >= lon_lo && lon <= lon_hi)
        {
            sum += aods[i];
            ++cnt;
        }
    }

    /* R's mean() on an empty vector returns NaN */
    means[g] = (cnt > 0) ? (sum / cnt)
                         : __longlong_as_double(0x7FF8000000000000LL);
}

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */
static void cuda_check(cudaError_t err, const char* ctx)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", ctx, cudaGetErrorString(err));
        exit(1);
    }
}

struct ImageEntry { int index; std::string path; };

static std::vector<ImageEntry> load_manifest(const char* path)
{
    std::vector<ImageEntry> v;
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open manifest: %s\n", path); exit(1); }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        ImageEntry e;
        ss >> e.index >> e.path;
        v.push_back(e);
    }
    return v;
}

/*
 * Read binary produced by preprocess.py.
 * Layout: int32 n | float64[n] lats | float64[n] lons | float64[n] aods
 */
static bool read_bin(const std::string& path,
                     std::vector<double>& lats,
                     std::vector<double>& lons,
                     std::vector<double>& aods)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { fprintf(stderr, "  Cannot open %s\n", path.c_str()); return false; }
    int n = 0;
    if (fread(&n, sizeof(int), 1, fp) != 1 || n <= 0) { fclose(fp); return false; }
    lats.resize(n); lons.resize(n); aods.resize(n);
    bool ok = (fread(lats.data(), sizeof(double), n, fp) == (size_t)n) &&
              (fread(lons.data(), sizeof(double), n, fp) == (size_t)n) &&
              (fread(aods.data(), sizeof(double), n, fp) == (size_t)n);
    fclose(fp);
    return ok;
}

/* Build cell-centre coordinate arrays (double precision, matching R) */
static void build_grid_centers(double* out_lon, double* out_lat)
{
    for (int lat_i = 0; lat_i < N_LAT; ++lat_i) {
        double lat_c = (lat_i + 0.5) * RESOLUTION + LAT_MIN;
        for (int lon_i = 0; lon_i < N_LON; ++lon_i) {
            double lon_c = (lon_i + 0.5) * RESOLUTION + LON_MIN;
            int g = lat_i * N_LON + lon_i;
            out_lon[g] = lon_c;
            out_lat[g] = lat_c;
        }
    }
}

/* Write CSV — same columns as R's data.frame: long, lat, AOD */
static bool write_csv(const std::string& path,
                      const double* lons,
                      const double* lats,
                      const double* means,
                      int n_grid)
{
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) { fprintf(stderr, "  Cannot write %s\n", path.c_str()); return false; }
    fprintf(fp, "long,lat,AOD\n");
    for (int i = 0; i < n_grid; ++i) {
        if (std::isnan(means[i]))
            fprintf(fp, "%.8f,%.8f,NaN\n", lons[i], lats[i]);
        else
            fprintf(fp, "%.8f,%.8f,%.17g\n", lons[i], lats[i], means[i]);
    }
    fclose(fp);
    return true;
}

/* -------------------------------------------------------------------------
 * Callable module entry point
 *
 * Launches k_regrid for one image already on the GPU.
 * Caller must pre-allocate d_means (N_GRID doubles on device).
 * Synchronizes before returning.
 * ---------------------------------------------------------------------- */
void regrid_gpu(const double* d_lats,
                const double* d_lons,
                const double* d_aods,
                int           n_points,
                double*       d_means)
{
    int threads = 256;
    int blocks  = (N_GRID + threads - 1) / threads;
    k_regrid<<<blocks, threads>>>(d_lats, d_lons, d_aods, n_points, d_means);
    cuda_check(cudaGetLastError(), "k_regrid launch");
    cuda_check(cudaDeviceSynchronize(), "regrid sync");
}

/* -------------------------------------------------------------------------
 * Standalone verifier (only compiled with -DREGRID_STANDALONE_MAIN)
 * ---------------------------------------------------------------------- */
#ifdef REGRID_STANDALONE_MAIN
int main(int argc, char* argv[])
{
    const char* manifest_path = (argc > 1) ? argv[1] : "manifest_G16.txt";
    const char* csv_dir       = (argc > 2) ? argv[2] : "output/G16";

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", csv_dir);
    (void)system(cmd);

    printf("GPU regrid — exact replica of regrid.R\n");
    printf("Grid: N_LON=%d  N_LAT=%d  N_GRID=%d\n", N_LON, N_LAT, N_GRID);
    printf("LON [%.2f, %.2f]  LAT [%.2f, %.2f]  res=%.4f\n",
           LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, RESOLUTION);
    printf("Bounds: inclusive (>= and <=), matching R's subset() call\n\n");

    /* GPU info */
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0), "getDeviceProperties");
    printf("GPU: %s  (compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

    /* Pre-compute cell centres */
    static double grid_lons[N_GRID], grid_lats[N_GRID];
    build_grid_centers(grid_lons, grid_lats);

    /* Allocate GPU buffers */
    double *d_lats = nullptr, *d_lons = nullptr, *d_aods = nullptr;
    double *d_means = nullptr;
    int     alloc_pts = 0;

    cuda_check(cudaMalloc(&d_means, N_GRID * sizeof(double)), "malloc means");

    /* Host buffer for output */
    static double h_means[N_GRID];

    auto manifest = load_manifest(manifest_path);
    printf("Processing %zu images ...\n", manifest.size());

    double t_h2d = 0, t_kernel = 0, t_d2h = 0, t_io = 0;
    double t_wall_start = now_ms();

    for (const auto& img : manifest) {
        printf("  [%02d] ", img.index);
        fflush(stdout);

        double t0;
        std::vector<double> h_lats, h_lons, h_aods;
        t0 = now_ms();
        if (!read_bin(img.path, h_lats, h_lons, h_aods)) {
            printf("SKIP (read error)\n");
            continue;
        }
        t_io += now_ms() - t0;

        int n_pts = (int)h_lats.size();
        printf("%d pts  ", n_pts);
        fflush(stdout);

        /* Grow input buffers if needed */
        if (n_pts > alloc_pts) {
            if (d_lats) { cudaFree(d_lats); cudaFree(d_lons); cudaFree(d_aods); }
            cuda_check(cudaMalloc(&d_lats, n_pts * sizeof(double)), "malloc lats");
            cuda_check(cudaMalloc(&d_lons, n_pts * sizeof(double)), "malloc lons");
            cuda_check(cudaMalloc(&d_aods, n_pts * sizeof(double)), "malloc aods");
            alloc_pts = n_pts;
        }

        /* Copy to GPU */
        t0 = now_ms();
        cuda_check(cudaMemcpy(d_lats, h_lats.data(), n_pts*sizeof(double),
                              cudaMemcpyHostToDevice), "H2D lats");
        cuda_check(cudaMemcpy(d_lons, h_lons.data(), n_pts*sizeof(double),
                              cudaMemcpyHostToDevice), "H2D lons");
        cuda_check(cudaMemcpy(d_aods, h_aods.data(), n_pts*sizeof(double),
                              cudaMemcpyHostToDevice), "H2D aods");
        t_h2d += now_ms() - t0;

        /* Launch: one thread per output cell */
        t0 = now_ms();
        int threads = 256;
        int blocks  = (N_GRID + threads - 1) / threads;  /* = 15 blocks */
        k_regrid<<<blocks, threads>>>(d_lats, d_lons, d_aods, n_pts, d_means);
        cuda_check(cudaGetLastError(), "k_regrid launch");
        cuda_check(cudaDeviceSynchronize(), "sync");
        t_kernel += now_ms() - t0;

        /* Copy result */
        t0 = now_ms();
        cuda_check(cudaMemcpy(h_means, d_means, N_GRID*sizeof(double),
                              cudaMemcpyDeviceToHost), "D2H means");
        t_d2h += now_ms() - t0;

        /* Write CSV */
        t0 = now_ms();
        char csv_path[512];
        snprintf(csv_path, sizeof(csv_path), "%s/img_%03d.csv", csv_dir, img.index);
        if (write_csv(csv_path, grid_lons, grid_lats, h_means, N_GRID))
            printf("-> %s\n", csv_path);
        else
            printf("WRITE ERROR\n");
        t_io += now_ms() - t0;
    }

    cudaFree(d_means);
    if (d_lats) { cudaFree(d_lats); cudaFree(d_lons); cudaFree(d_aods); }

    double t_wall = now_ms() - t_wall_start;
    int    n_imgs = (int)manifest.size();

    printf("\n%s\n", std::string(60,'=').c_str());
    printf("GPU REGRID TIMING SUMMARY\n");
    printf("%s\n", std::string(60,'=').c_str());
    printf("  Images processed     : %d\n", n_imgs);
    printf("  H2D transfers        : %8.2f ms total  (%6.3f ms/img)\n",
           t_h2d,   t_h2d   / n_imgs);
    printf("  GPU kernel execution : %8.2f ms total  (%6.3f ms/img)\n",
           t_kernel, t_kernel / n_imgs);
    printf("  D2H transfers        : %8.2f ms total  (%6.3f ms/img)\n",
           t_d2h,   t_d2h   / n_imgs);
    printf("  File I/O (bin+csv)   : %8.2f ms total  (%6.3f ms/img)\n",
           t_io,    t_io    / n_imgs);
    printf("  Wall time (total)    : %8.2f ms        (%6.3f ms/img)\n",
           t_wall,  t_wall  / n_imgs);
    printf("%s\n", std::string(60,'=').c_str());
    printf("\nDone. CSV files in: %s\n", csv_dir);
    return 0;
}
#endif /* REGRID_STANDALONE_MAIN */
