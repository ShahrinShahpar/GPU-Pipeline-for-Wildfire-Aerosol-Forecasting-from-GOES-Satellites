#!/usr/bin/env python3
"""
14_plots.py — Generate all GPU result figures matching R's application.R style.

Individual R-style GPU figures (one per timestep, named like R's output):
  fig_9_row_c_time5/10/15/20    — Filtered AOD field (viridis "D", 0–3.5)
  fig_9_row_d_time5/10/15/20    — Bias field (plasma, −2.5–2.5)
  fig_10a                        — Wind velocity field
  fig_10b                        — Diffusivity K field
  fig_11_row_b_time21/22/23/24  — G17 k-step forecast error (plasma, −3–3)
  fig_11_row_d_time21/22/23/24  — G16 k-step forecast error (plasma, −3–3)
  fig_14a/b/c                    — Prediction-difference histograms

Comparison figures (GPU vs R, kept from before):
  fig9c_filtered_aod, fig9d_bias_field, rmse_over_time, forecast_kstep, coef_scatter
"""

import os, sys, csv, math, warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# ── paths ──────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.join(script_dir, '..', '..')
out_dir    = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo_root, 'output')
EXP = os.path.join(repo_root, 'reference', 'exports')

PLOTS_DIR = os.path.join(out_dir, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

GPU_MFLT_GIBBS = os.path.join(out_dir, 'G16_gibbs', 'm_flt_gibbs.csv')
GPU_MFLT_EXTRA = os.path.join(out_dir, 'G16_gibbs', 'm_flt.csv')
GPU_G          = os.path.join(out_dir, 'G16_g',     'G.csv')
R_MFLT         = os.path.join(EXP, 'm_flt.csv')
OMEGA1         = os.path.join(EXP, 'Omega1.csv')
OMEGA2         = os.path.join(EXP, 'Omega2.csv')
MSE_CSV        = os.path.join(out_dir, 'G16_gibbs', 'mse_summary.csv')
G17_RAW        = os.path.join(repo_root, 'output', 'G17_processed.csv')
G16_RAW        = os.path.join(repo_root, 'output', 'G16_processed.csv')
V_A_CSV        = os.path.join(EXP, 'v_a.csv')
K_IFM_CSV      = os.path.join(EXP, 'K_ifm.csv')
DIF1716_CSV    = os.path.join(EXP, 'dif1716.csv')
DIF16_CSV      = os.path.join(EXP, 'dif16.csv')
DIF17_CSV      = os.path.join(EXP, 'dif17.csv')

Nr = 60;  N = 20;  N2 = N*N;  T = 20
LON0, LAT0, GSTEP = -124.0, 35.0, 0.04
lon_vals = LON0 + np.arange(Nr) * GSTEP
lat_vals = LAT0 + np.arange(Nr) * GSTEP
LON_FLAT = np.tile(lon_vals, Nr)
LAT_FLAT = np.repeat(lat_vals, Nr)

# ── R's theme_paper ────────────────────────────────────────────────────────
def apply_theme_paper(ax):
    """Mimic R's theme_paper: white bg, black border, white grid."""
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)
    ax.grid(True, color='white', linewidth=0.5)
    ax.tick_params(labelsize=11)

# ── county boundaries ──────────────────────────────────────────────────────
def add_county_lines(ax):
    """Overlay California county boundaries if cartopy is available."""
    try:
        import cartopy.feature as cfeature
        import cartopy.crs as ccrs
        counties = cfeature.NaturalEarthFeature(
            category='cultural', name='admin_2_counties',
            scale='10m', facecolor='none', edgecolor='black', linewidth=0.5)
        ax.add_feature(counties)
    except Exception:
        pass   # cartopy not available or counties not found — skip silently

# ── loaders ────────────────────────────────────────────────────────────────
def load_omega(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((float(row['k1']), float(row['k2'])))
    return rows

def build_F(omega1, omega2, sites_lon, sites_lat):
    x = (sites_lon - LON0) / (Nr * GSTEP)
    y = (sites_lat - LAT0) / (Nr * GSTEP)
    F1_cols  = [np.cos(2*math.pi*(k1*x + k2*y))   for k1,k2 in omega1]
    F2c_cols = [2*np.cos(2*math.pi*(k1*x + k2*y)) for k1,k2 in omega2]
    F2s_cols = [2*np.sin(2*math.pi*(k1*x + k2*y)) for k1,k2 in omega2]
    return np.column_stack(F1_cols + F2c_cols + F2s_cols)

def load_mflt(path, n_coef=800):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            t   = int(row['time_step'])
            idx = int(row['coef_idx']) - 1
            val = float(row['value'])
            if t not in data:
                data[t] = np.zeros(n_coef)
            if idx < n_coef:
                data[t][idx] = val
    return data

def load_G(path):
    rows = []
    with open(path) as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            rows.append([float(v) for v in row])
    return np.array(rows)

def load_raw_aod(path, time_index):
    """Load full 3600-pixel AOD at a given time_index. Returns (lon, lat, aod) arrays."""
    lons, lats, aods = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['time_index']) == time_index:
                lons.append(float(row['long']))
                lats.append(float(row['lat']))
                v = row['AOD']
                aods.append(float('nan') if v.strip() in ('', 'NaN', 'NA', 'nan') else float(v))
    return np.array(lons), np.array(lats), np.array(aods)

def load_flat_csv(path, col):
    vals = []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row[col]
            vals.append(float('nan') if v.strip() in ('', 'NaN', 'NA', 'nan') else float(v))
    return np.array(vals)

def load_dif_csv(path):
    vals = []
    with open(path) as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            for v in row:
                try: vals.append(float(v))
                except: pass
    return np.array(vals)

def load_mse_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            t = int(row['time_step'])
            data[t] = {k: float(v) for k, v in row.items() if k != 'time_step'}
    return data

def aod_field(F, theta):
    return (F @ theta[:N2]).reshape(Nr, Nr)

def bias_field(F, theta):
    return (F @ theta[N2:2*N2]).reshape(Nr, Nr)

# ── load data ──────────────────────────────────────────────────────────────
print("Loading omega sets …")
omega1 = load_omega(OMEGA1)
omega2 = load_omega(OMEGA2)

print("Building F matrix (3600 × 400) …")
F = build_F(omega1, omega2, LON_FLAT, LAT_FLAT)

print("Loading m_flt (GPU Gibbs last iter + extra pass, and R) …")
# GPU: m_flt_gibbs = last Gibbs iteration's forward filter (matches R's returned m.flt exactly)
gpu_mflt = load_mflt(GPU_MFLT_GIBBS)
# Extra pass for scatter comparison only
gpu_mflt_extra = load_mflt(GPU_MFLT_EXTRA) if os.path.exists(GPU_MFLT_EXTRA) else gpu_mflt

r_mflt_raw = load_mflt(R_MFLT)
r_mflt = {t: r_mflt_raw[t] for t in range(1, T+1) if t in r_mflt_raw}

print("Loading G matrix …")
G = load_G(GPU_G)

print("Loading MSE summary …")
mse = load_mse_csv(MSE_CSV)

# ── load velocity, diffusivity ─────────────────────────────────────────────
has_fig10 = os.path.exists(V_A_CSV) and os.path.exists(K_IFM_CSV)
if has_fig10:
    print("Loading v_a and K_ifm …")
    v_lon  = load_flat_csv(V_A_CSV, 'long')
    v_lat  = load_flat_csv(V_A_CSV, 'lat')
    v_x    = load_flat_csv(V_A_CSV, 'v_x')
    v_y    = load_flat_csv(V_A_CSV, 'v_y')
    # speed in km/h matching R: speed.avg.smooth*60*0.04*111*12
    speed_raw = np.sqrt(v_x**2 + v_y**2)
    angle     = np.arctan2(v_y, v_x)
    speed_kmh = speed_raw * 60 * 0.04 * 111 * 12

    k_lon  = load_flat_csv(K_IFM_CSV, 'long')
    k_lat  = load_flat_csv(K_IFM_CSV, 'lat')
    k_vals = load_flat_csv(K_IFM_CSV, 'K')
    k_kmh  = k_vals * 60*60*0.04*111*0.04*111*12   # diffusivity km²/h

# ── load dif matrices for fig_14 ──────────────────────────────────────────
has_fig14 = (os.path.exists(DIF1716_CSV) and os.path.exists(DIF16_CSV)
             and os.path.exists(DIF17_CSV))
if has_fig14:
    print("Loading dif matrices …")
    dif1716 = load_dif_csv(DIF1716_CSV)
    dif16   = load_dif_csv(DIF16_CSV)
    dif17   = load_dif_csv(DIF17_CSV)

print("All data loaded.\n")

# ══════════════════════════════════════════════════════════════════════════
# HELPER: R-style single-panel raster map
# ══════════════════════════════════════════════════════════════════════════
def save_raster_map(data_2d, lon_ax, lat_ax, cmap, vmin, vmax,
                    title, cbar_label, out_path, na_color='white',
                    width_cm=10, height_cm=8):
    """Save one raster map in R's theme_paper style (white bg, black border)."""
    fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54))
    apply_theme_paper(ax)
    masked = np.where(np.isnan(data_2d), np.nan, data_2d)
    import copy
    cmap_obj = copy.copy(plt.cm.get_cmap(cmap))
    cmap_obj.set_bad(color=na_color)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(lon_ax, lat_ax, masked,
                       cmap=cmap_obj, norm=norm, shading='auto')
    add_county_lines(ax)
    ax.set_xlim(-124, -121.6);  ax.set_ylim(35.0, 37.4)
    ax.set_xlabel('long', fontsize=14);  ax.set_ylabel('lat', fontsize=14)
    ax.set_title(title, fontsize=18, ha='center')
    ax.tick_params(labelsize=11)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cbar_label, fontsize=14)
    cb.ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

# ══════════════════════════════════════════════════════════════════════════
# Fig 9c — Filtered AOD field, individual R-style maps
# R: scale_fill_viridis_c(option="D", limits=c(NA,3.5))  → viridis, no lower clip
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig_9_row_c …")
TIME_IDS  = [5, 10, 15, 20]
TIME_LBLS = ['time 5', 'time 10', 'time 15', 'time 20']
TIME_SAVE = ['time5',  'time10',  'time15',  'time20']

for t, lbl, sav in zip(TIME_IDS, TIME_LBLS, TIME_SAVE):
    if t not in gpu_mflt:
        print(f"  skip t={t} (no GPU data)")
        continue
    aod = aod_field(F, gpu_mflt[t])
    vmin = float(np.nanmin(aod))   # R's limits=c(NA, 3.5) → no lower clip
    out  = os.path.join(PLOTS_DIR, f'fig_9_row_c_{sav}.png')
    save_raster_map(aod, lon_vals, lat_vals, 'viridis', vmin, 3.5,
                    lbl, 'AOD', out)

# ══════════════════════════════════════════════════════════════════════════
# Fig 9d — Bias field α₂, individual R-style maps
# R: scale_fill_viridis_c(option="plasma", limits=c(-2.5, 2.5))
# ══════════════════════════════════════════════════════════════════════════
print("Generating fig_9_row_d …")
for t, lbl, sav in zip(TIME_IDS, TIME_LBLS, TIME_SAVE):
    if t not in gpu_mflt:
        continue
    bias = bias_field(F, gpu_mflt[t])
    out  = os.path.join(PLOTS_DIR, f'fig_9_row_d_{sav}.png')
    save_raster_map(bias, lon_vals, lat_vals, 'plasma', -2.5, 2.5,
                    lbl, 'bias', out)

# ══════════════════════════════════════════════════════════════════════════
# Fig 10a — Wind velocity
# R: scale_fill_viridis_c(option="D", limits=c(0,30)), geom_spoke for arrows
# ══════════════════════════════════════════════════════════════════════════
if has_fig10:
    print("Generating fig_10a …")
    spd_grid   = speed_kmh.reshape(Nr, Nr)
    angle_grid = angle.reshape(Nr, Nr)
    fig, ax = plt.subplots(figsize=(15/2.54, 11/2.54))
    apply_theme_paper(ax)
    im = ax.pcolormesh(lon_vals, lat_vals, spd_grid,
                       cmap='viridis', vmin=0, vmax=30, shading='auto')
    # arrow subsampling (every 3rd pixel, matching R's seq(1,60,3))
    sub = slice(None, None, 3)
    lon_sub = lon_vals[sub]
    lat_sub = lat_vals[sub]
    spd_sub = spd_grid[sub, :][:, sub]        # speed in km/h (for color only)
    raw_sub = speed_raw.reshape(Nr, Nr)[sub, :][:, sub]   # raw speed for radius scaling
    ang_sub = angle.reshape(Nr, Nr)[sub, :][:, sub]
    # R: radius = scales::rescale(speed, c(0.05, 0.1))  — arrow length in degrees
    s_min, s_max = raw_sub.min(), raw_sub.max()
    denom = (s_max - s_min) if s_max > s_min else 1.0
    radius = 0.05 + (raw_sub - s_min) / denom * 0.05    # [0.05, 0.10] degrees
    U = radius * np.cos(ang_sub)
    V = radius * np.sin(ang_sub)
    ax.quiver(lon_sub, lat_sub, U, V,
              angles='xy', scale_units='xy', scale=1,
              pivot='tail', width=0.003, color='black', alpha=0.8)
    add_county_lines(ax)
    ax.set_xlim(-124, -121.6);  ax.set_ylim(35.0, 37.4)
    ax.set_xlabel('long', fontsize=14);  ax.set_ylabel('lat', fontsize=14)
    ax.set_title('(a)', fontsize=18, ha='center')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('speed: km/h', fontsize=14);  cb.ax.tick_params(labelsize=11)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'fig_10a.png')
    plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
    print(f'Saved: {out}')

    print("Generating fig_10b …")
    k_grid = k_kmh.reshape(Nr, Nr)
    out = os.path.join(PLOTS_DIR, 'fig_10b.png')
    save_raster_map(k_grid, lon_vals, lat_vals, 'viridis',
                    float(np.nanmin(k_grid)), float(np.nanmax(k_grid)),
                    '(b)', 'diffusivity: km²/h', out,
                    width_cm=16, height_cm=11)

# ══════════════════════════════════════════════════════════════════════════
# Fig 11 — K-step forecast errors
# R: F @ G^k @ m.flt[[21]][1:N^2]  −  G17/G16.aod.raw[[40]]$AOD
#    (R has a bug: uses i=20 → t=40 for ALL k instead of t=20+k)
#    plasma, limits=c(-3, 3), na.value="white"
# Use KNN-imputed data (no NaN) for the observation so the full spatial
# field is visible — this eliminates white gaps from cloud-masked pixels.
# ══════════════════════════════════════════════════════════════════════════

def load_knn_img(knn_dir, t_idx):
    """Load a 60×60 KNN CSV (header row V1..V60) and return a flat (3600,) array."""
    path = os.path.join(knn_dir, f'img_{t_idx:03d}.csv')
    if not os.path.exists(path):
        return None
    mat = np.genfromtxt(path, delimiter=',', skip_header=1)   # (60,60)
    return mat.flatten()                                        # row-major → lat×lon order

G17_KNN_DIR = os.path.join(out_dir, 'G17_knn')
G16_KNN_DIR = os.path.join(out_dir, 'G16_knn')

print("Loading KNN-imputed AOD at t=40 for fig_11 …")
g17_knn40 = load_knn_img(G17_KNN_DIR, 40)
g16_knn40 = load_knn_img(G16_KNN_DIR, 40)

if g17_knn40 is not None and 20 in gpu_mflt:
    print("Generating fig_11_row_b (G17 k-step forecast error) …")
    alpha1_t20 = gpu_mflt[T][:N2]
    Gk_alpha = alpha1_t20.copy()
    for k in range(1, 5):
        Gk_alpha = G @ Gk_alpha
        pred_flat = F @ Gk_alpha                   # (3600,) full field, no NaN
        diff_flat = pred_flat - g17_knn40           # KNN obs: no NaN → no white gaps
        diff_2d   = diff_flat.reshape(Nr, Nr)
        time_lbl  = f'time {20+k}'
        time_sav  = f'time{20+k}'
        out = os.path.join(PLOTS_DIR, f'fig_11_row_b_{time_sav}.png')
        save_raster_map(diff_2d, lon_vals, lat_vals, 'plasma', -3, 3,
                        time_lbl, 'AOD', out)

if g16_knn40 is not None and 20 in gpu_mflt:
    print("Generating fig_11_row_d (G16 k-step forecast error) …")
    alpha1_t20 = gpu_mflt[T][:N2]
    Gk_alpha = alpha1_t20.copy()
    for k in range(1, 5):
        Gk_alpha = G @ Gk_alpha
        pred_flat = F @ Gk_alpha
        diff_flat = pred_flat - g16_knn40
        diff_2d   = diff_flat.reshape(Nr, Nr)
        time_lbl  = f'time {20+k}'
        time_sav  = f'time{20+k}'
        out = os.path.join(PLOTS_DIR, f'fig_11_row_d_{time_sav}.png')
        save_raster_map(diff_2d, lon_vals, lat_vals, 'plasma', -3, 3,
                        time_lbl, 'AOD', out)

# ══════════════════════════════════════════════════════════════════════════
# Fig 11 COMPARISON — GPU vs R forecast error (3-row: GPU / R / difference)
#   Same KNN observation used for both to isolate prediction differences.
#   Uses GPU G matrix for both so only alpha1 differences drive the gap.
# ══════════════════════════════════════════════════════════════════════════
_have_r_for11 = (T in r_mflt)
for sat_tag, obs_knn40 in [('b_G17', g17_knn40), ('d_G16', g16_knn40)]:
    if obs_knn40 is None or not _have_r_for11:
        continue
    print(f"Generating fig_11_comparison_{sat_tag} (GPU vs R forecast error) …")
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(
        f'Fig 11 row {"b" if "b" in sat_tag else "d"} — K-step Forecast Error vs KNN obs t=40'
        f'  (GPU top / R middle / diff bottom)',
        fontsize=12, fontweight='bold'
    )
    alpha_gpu = gpu_mflt[T][:N2].copy()
    alpha_r   = r_mflt[T][:N2].copy()
    Gk_gpu = alpha_gpu; Gk_r = alpha_r
    for col, k in enumerate(range(1, 5)):
        Gk_gpu = G @ Gk_gpu;  Gk_r = G @ Gk_r
        err_gpu = (F @ Gk_gpu - obs_knn40).reshape(Nr, Nr)
        err_r   = (F @ Gk_r   - obs_knn40).reshape(Nr, Nr)
        err_dif = err_gpu - err_r
        for row, (data, title, cmap, vlo, vhi) in enumerate([
            (err_gpu, f'GPU  k={k}', 'plasma', -3,    3   ),
            (err_r,   f'R    k={k}', 'plasma', -3,    3   ),
            (err_dif, f'GPU−R k={k}','RdBu_r', -1.5,  1.5 ),
        ]):
            ax = axes[row, col]
            im = ax.pcolormesh(lon_vals, lat_vals, data, cmap=cmap,
                               vmin=vlo, vmax=vhi, shading='auto')
            ax.set_xlim(-124, -121.6);  ax.set_ylim(35.0, 37.4)
            ax.set_aspect('equal');  ax.tick_params(labelsize=7)
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'fig11_comparison_{sat_tag}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
    print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════
# Fig 14 — Prediction-difference histograms
# R: hist(dif, breaks=100, freq=FALSE, xlab="prediction difference", main="(a)")
# ══════════════════════════════════════════════════════════════════════════
if has_fig14:
    print("Generating fig_14a/b/c …")
    for dif, title, fname in [
        (dif1716, '(a)', 'fig_14a.png'),
        (dif16,   '(b)', 'fig_14b.png'),
        (dif17,   '(c)', 'fig_14c.png'),
    ]:
        dif_clean = dif[~np.isnan(dif)]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(0.8)
        ax.hist(dif_clean, bins=100, density=True, color='gray', edgecolor='gray')
        ax.set_xlabel('prediction difference', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title(title, fontsize=18, ha='center')
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, fname)
        plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
        print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════
# COMPARISON: Fig 9c GPU vs R side-by-side (keep existing)
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating comparison panels …")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Filtered AOD Field  —  GPU (top) vs R (bottom)', fontsize=14, fontweight='bold')
for col, (t, lbl) in enumerate(zip(TIME_IDS, TIME_LBLS)):
    for row, (mflt_dict, label) in enumerate([(gpu_mflt, 'GPU'), (r_mflt, 'R')]):
        ax = axes[row, col]
        if t not in mflt_dict:
            ax.set_visible(False); continue
        aod = aod_field(F, mflt_dict[t])
        im  = ax.pcolormesh(lon_vals, lat_vals, aod, cmap='viridis',
                            vmin=0, vmax=3.5, shading='auto')
        ax.set_xlim(-124, -121.6);  ax.set_ylim(35.0, 37.4)
        ax.set_aspect('equal');  ax.tick_params(labelsize=7)
        if row == 0: ax.set_title(lbl, fontsize=10)
        if col == 0: ax.set_ylabel(label, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8, label='AOD' if col == 3 else '')
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'fig9c_filtered_aod.png')
plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
print(f'Saved: {out}')

# COMPARISON: Fig 9d GPU vs R
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Bias Field (α₂)  —  GPU (top) vs R (bottom)', fontsize=14, fontweight='bold')
for col, (t, lbl) in enumerate(zip(TIME_IDS, TIME_LBLS)):
    for row, (mflt_dict, label) in enumerate([(gpu_mflt, 'GPU'), (r_mflt, 'R')]):
        ax = axes[row, col]
        if t not in mflt_dict:
            ax.set_visible(False); continue
        bias = bias_field(F, mflt_dict[t])
        im   = ax.pcolormesh(lon_vals, lat_vals, bias, cmap='plasma',
                             vmin=-2.5, vmax=2.5, shading='auto')
        ax.set_xlim(-124, -121.6);  ax.set_ylim(35.0, 37.4)
        ax.set_aspect('equal');  ax.tick_params(labelsize=7)
        if row == 0: ax.set_title(lbl, fontsize=10)
        if col == 0: ax.set_ylabel(label, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Bias' if col == 3 else '')
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'fig9d_bias_field.png')
plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════
# RMSE over time
# ══════════════════════════════════════════════════════════════════════════
times   = sorted(mse.keys())
gpu_g17 = [mse[t]['rmse_gpu_g17'] for t in times]
r_g17   = [mse[t]['rmse_r_g17']   for t in times]
gpu_g16 = [mse[t]['rmse_gpu_g16'] for t in times]
r_g16   = [mse[t]['rmse_r_g16']   for t in times]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('In-Sample Filter RMSE Over Time', fontsize=13, fontweight='bold')
for ax, (g_vals, r_vals, sat) in zip(axes,
        [(gpu_g17, r_g17, 'G17 (GOES-17)'), (gpu_g16, r_g16, 'G16 (GOES-16)')]):
    ax.plot(times, g_vals, 'o-', color='steelblue', label='GPU', linewidth=2, markersize=4)
    ax.plot(times, r_vals, 's--', color='tomato',   label='R',   linewidth=2, markersize=4)
    ax.set_xlabel('Time step', fontsize=11);  ax.set_ylabel('RMSE (AOD units)', fontsize=11)
    ax.set_title(sat, fontsize=11);  ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3);  ax.set_xticks(times[::2])
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'rmse_over_time.png')
plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════
# K-step forecast comparison (GPU vs R)
# ══════════════════════════════════════════════════════════════════════════
theta_gpu_20 = gpu_mflt[T][:N2]
theta_r_20   = r_mflt[T][:N2]
Gk_gpu = theta_gpu_20.copy()
Gk_r   = theta_r_20.copy()

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle(f'K-Step AOD Forecast from t={T}  (GPU top / R middle / difference bottom)',
             fontsize=13, fontweight='bold')
for col, k in enumerate(range(1, 6)):
    Gk_gpu = G @ Gk_gpu;  Gk_r = G @ Gk_r
    pred_gpu = (F @ Gk_gpu).reshape(Nr, Nr)
    pred_r   = (F @ Gk_r).reshape(Nr, Nr)
    diff     = pred_gpu - pred_r
    vmax = max(np.nanpercentile(np.abs(pred_gpu), 98), np.nanpercentile(np.abs(pred_r), 98))
    for row, (data, title, cmap, vlim) in enumerate([
        (pred_gpu, f'GPU k={k}', 'viridis', (0, vmax)),
        (pred_r,   f'R   k={k}', 'viridis', (0, vmax)),
        (diff,     f'GPU−R k={k}','RdBu_r', (-0.5, 0.5)),
    ]):
        ax = axes[row, col]
        im = ax.pcolormesh(lon_vals, lat_vals, data, cmap=cmap,
                           vmin=vlim[0], vmax=vlim[1], shading='auto')
        ax.set_xlim(-124, -121.6);  ax.set_ylim(35.0, 37.4)
        ax.set_aspect('equal');  ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.75)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'forecast_kstep.png')
plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════
# Coefficient scatter GPU vs R
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('GPU vs R Filtered State Coefficients (α₁, first 400)', fontsize=12, fontweight='bold')
for ax, t in zip(axes, TIME_IDS):
    if t not in gpu_mflt_extra or t not in r_mflt:
        ax.set_visible(False); continue
    g = gpu_mflt_extra[t][:N2];  r = r_mflt[t][:N2]
    lim = max(np.abs(g).max(), np.abs(r).max()) * 1.05
    ax.scatter(r, g, s=6, alpha=0.5, color='steelblue')
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
    ax.set_xlabel('R coefficients', fontsize=9);  ax.set_ylabel('GPU coefficients', fontsize=9)
    ax.set_title(f't = {t}', fontsize=10)
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    corr = float(np.corrcoef(g, r)[0, 1])
    ax.text(0.05, 0.92, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'coef_scatter.png')
plt.savefig(out, dpi=300, bbox_inches='tight');  plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════
print(f'\nAll plots saved to: {PLOTS_DIR}')
print('  R-style individual maps:')
print('    fig_9_row_c_time5/10/15/20.png  — Filtered AOD (viridis, 0–3.5)')
print('    fig_9_row_d_time5/10/15/20.png  — Bias field (plasma, −2.5–2.5)')
if has_fig10:
    print('    fig_10a.png, fig_10b.png        — Velocity, diffusivity')
if g17_knn40 is not None:
    print('    fig_11_row_b_time21-24.png      — G17 k-step forecast error (plasma, −3–3)')
    print('    fig11_comparison_b_G17.png      — G17 forecast error GPU vs R (3-row)')
if g16_knn40 is not None:
    print('    fig_11_row_d_time21-24.png      — G16 k-step forecast error (plasma, −3–3)')
    print('    fig11_comparison_d_G16.png      — G16 forecast error GPU vs R (3-row)')
if has_fig14:
    print('    fig_14a/b/c.png                 — Prediction difference histograms')
print('  Comparison panels (GPU vs R):')
print('    fig9c_filtered_aod.png, fig9d_bias_field.png')
print('    rmse_over_time.png, forecast_kstep.png, coef_scatter.png')
