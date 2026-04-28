#!/usr/bin/env python3
"""
13_check_mse.py — MSE / RMSE validation for the Gibbs FFBS2 output.

Replicates what R's application.R does after the Gibbs sampler:

  Part A — In-sample filter RMSE
    For each t=1..20:
      pred_t = F1t[t] @ theta_t[0:N²]          (GPU filtered AOD at observed pixels)
      RMSE_t = sqrt(mean((pred_t - y1c_t)²))
    Done separately for GPU and R m_flt; both should give similar RMSE.

  Part B — K-step forecast comparison (GPU vs R)
    From the final filtered state at t=20:
      theta_20 = m_flt[t=20][0:N²]
      for k = 1..5:
        pred_k = F1t_20 @ G^k @ theta_20   (predicted AOD k steps ahead)
    GPU vs R absolute difference statistics — analogous to dif16.csv in R.

Usage:
  python3 validation/gibbs/13_check_mse.py <output_dir>

  <output_dir>  directory containing G16_gibbs/m_flt.csv and G16_g/G.csv
                (defaults to 'output')

Reference exports expected at:
  ../freshstaart/Physics-Informed-Statistical-Modeling-for-Wildfire-Aerosols-Process-main/data/exports/
"""

import sys
import os
import csv
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── paths ────────────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
repo_root   = os.path.join(script_dir, '..', '..')
out_dir     = sys.argv[1] if len(sys.argv) > 1 else 'output'
if not os.path.isabs(out_dir):
    out_dir = os.path.join(repo_root, out_dir)

EXP = os.path.join(repo_root, 'reference', 'exports')

GPU_MFLT  = os.path.join(out_dir, 'G16_gibbs', 'm_flt.csv')
GPU_G     = os.path.join(out_dir, 'G16_g',     'G.csv')
R_MFLT    = os.path.join(EXP, 'm_flt.csv')
F1T_DIR   = os.path.join(EXP, 'F1t')
F2T_DIR   = os.path.join(EXP, 'F2t')
Y1C_DIR   = os.path.join(EXP, 'y1c')
Y2C_DIR   = os.path.join(EXP, 'y2c')

T   = 20      # number of time steps
N2  = 400     # N^2 = 20^2 Fourier basis size


# ─── helpers ──────────────────────────────────────────────────────────────────
def load_mflt(path, n_coef=800):
    """Return dict: time_step (1-based int) -> list of n_coef floats."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t   = int(row['time_step'])
            idx = int(row['coef_idx']) - 1      # 0-based
            val = float(row['value'])
            if t not in data:
                data[t] = [0.0] * n_coef
            if idx < n_coef:
                data[t][idx] = val
    return data


def load_F1t(t):
    """Load dense F1t matrix for time step t. Returns list-of-list [n_obs][N2]."""
    path = os.path.join(F1T_DIR, 'F1t_t%02d.csv' % t)
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)                    # skip header (V1,V2,...)
        return [[float(x) for x in row] for row in reader]


def load_F2t(t):
    """Load dense F2t matrix for time step t. Returns list-of-list [n_obs][N2]."""
    path = os.path.join(F2T_DIR, 'F2t_t%02d.csv' % t)
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        return [[float(x) for x in row] for row in reader]


def load_y1c(t):
    """Return observed G17 AOD values for time step t (list of floats)."""
    path = os.path.join(Y1C_DIR, 'y1c_t%02d.csv' % t)
    vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals.append(float(row['AOD']))
    return vals


def load_y2c(t):
    """Return observed G16 AOD values for time step t (list of floats)."""
    path = os.path.join(Y2C_DIR, 'y2c_t%02d.csv' % t)
    vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals.append(float(row['AOD']))
    return vals


def load_G(path):
    """Load 400×400 G matrix from CSV (first row is header V1..V400)."""
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)            # skip header
        return [[float(x) for x in row] for row in reader]


def matvec(M, v):
    """Matrix-vector product: M (n×m) × v (m) -> result (n)."""
    n = len(M)
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(n)]


def matmat_vec(G, v, k):
    """Apply G^k to vector v."""
    result = v[:]
    for _ in range(k):
        result = matvec(G, result)
    return result


def mse(pred, obs):
    n = len(pred)
    return sum((pred[i] - obs[i]) ** 2 for i in range(n)) / n


def rmse(pred, obs):
    return math.sqrt(mse(pred, obs))


def mean_abs(v):
    return sum(abs(x) for x in v) / len(v)


# ─── load data ────────────────────────────────────────────────────────────────
if not os.path.exists(GPU_MFLT):
    print('ERROR: GPU m_flt not found:', GPU_MFLT)
    sys.exit(1)

print('Loading GPU m_flt ...')
gpu_mflt = load_mflt(GPU_MFLT)

print('Loading R m_flt   ...')
r_mflt_all = load_mflt(R_MFLT)
# R has t=0 (m0) and t=1..20; keep only t=1..20
r_mflt = {t: r_mflt_all[t] for t in range(1, T+1) if t in r_mflt_all}

print('Loading G matrix  ...')
G = load_G(GPU_G)

print()

# ─── Part A: In-sample filter RMSE ───────────────────────────────────────────
print('=' * 65)
print('Part A — In-sample filter RMSE (GPU vs R vs each other)')
print('=' * 65)
print('  t    RMSE_gpu(G17)  RMSE_r(G17)   RMSE_gpu(G16)  RMSE_r(G16)')
print('  ' + '-' * 62)

rmse_gpu_g17 = []
rmse_r_g17   = []
rmse_gpu_g16 = []
rmse_r_g16   = []

for t in range(1, T+1):
    F1t  = load_F1t(t)
    F2t  = load_F2t(t)
    y1c  = load_y1c(t)
    y2c  = load_y2c(t)

    theta_gpu = gpu_mflt[t][:N2]    # first 400 = G17 basis coefficients
    theta_r   = r_mflt[t][:N2]

    # G17 predictions
    pred_gpu_g17 = matvec(F1t, theta_gpu)
    pred_r_g17   = matvec(F1t, theta_r)

    # G16 prediction uses only the first N2 state coefficients (alpha1).
    # The combined F matrix is built as [F1t|0 ; F2t|0] — the second N2 block
    # (alpha2) is zero in the observation equation; it only affects state dynamics.
    pred_gpu_g16 = matvec(F2t, theta_gpu)
    pred_r_g16   = matvec(F2t, theta_r)

    rg17 = rmse(pred_gpu_g17, y1c)
    rr17 = rmse(pred_r_g17,   y1c)
    rg16 = rmse(pred_gpu_g16, y2c)
    rr16 = rmse(pred_r_g16,   y2c)

    rmse_gpu_g17.append(rg17)
    rmse_r_g17.append(rr17)
    rmse_gpu_g16.append(rg16)
    rmse_r_g16.append(rr16)

    print('  t=%02d   %10.6f     %10.6f     %10.6f     %10.6f' %
          (t, rg17, rr17, rg16, rr16))

avg_gpu_g17 = sum(rmse_gpu_g17) / T
avg_r_g17   = sum(rmse_r_g17)   / T
avg_gpu_g16 = sum(rmse_gpu_g16) / T
avg_r_g16   = sum(rmse_r_g16)   / T

print('  ' + '-' * 62)
print('  mean   %10.6f     %10.6f     %10.6f     %10.6f' %
      (avg_gpu_g17, avg_r_g17, avg_gpu_g16, avg_r_g16))
print()
print('  RMSE ratio GPU/R (G17): %.4f' % (avg_gpu_g17 / avg_r_g17 if avg_r_g17 else float('nan')))
print('  RMSE ratio GPU/R (G16): %.4f' % (avg_gpu_g16 / avg_r_g16 if avg_r_g16 else float('nan')))
print()

# ─── Part B: K-step forecast comparison (GPU vs R) ───────────────────────────
print('=' * 65)
print('Part B — K-step forecast comparison at t=20 (GPU M2 vs R M2)')
print('  Using F1t[t=20] (G17 observed pixels) as evaluation set')
print('=' * 65)
print('  k  mean|GPU-R|   max|GPU-R|   RMSE(GPU-R)  mean|GPU|  mean|R|')
print('  ' + '-' * 62)

F1t_20 = load_F1t(T)   # F rows for G17 obs at t=20

theta_gpu_20 = gpu_mflt[T][:N2]
theta_r_20   = r_mflt[T][:N2]

for k in range(1, 6):
    gk_theta_gpu = matmat_vec(G, theta_gpu_20, k)
    gk_theta_r   = matmat_vec(G, theta_r_20,   k)

    pred_gpu_k = matvec(F1t_20, gk_theta_gpu)
    pred_r_k   = matvec(F1t_20, gk_theta_r)

    diffs = [abs(pred_gpu_k[i] - pred_r_k[i]) for i in range(len(pred_gpu_k))]
    mean_diff  = mean_abs(diffs)
    max_diff   = max(diffs)
    rmse_diff  = math.sqrt(sum(d**2 for d in diffs) / len(diffs))
    mg = mean_abs(pred_gpu_k)
    mr = mean_abs(pred_r_k)

    print('  k=%d  %10.6f   %10.6f   %10.6f    %10.6f %10.6f' %
          (k, mean_diff, max_diff, rmse_diff, mg, mr))

print()

# ─── Part C: MSE summary for reporting ───────────────────────────────────────
print('=' * 65)
print('Part C — Overall summary')
print('=' * 65)
total_mse_gpu = sum(r**2 for r in rmse_gpu_g17) / T
total_mse_r   = sum(r**2 for r in rmse_r_g17)   / T
print('  G17 filter MSE: GPU = %.6f   R = %.6f' % (total_mse_gpu, total_mse_r))

total_mse_gpu16 = sum(r**2 for r in rmse_gpu_g16) / T
total_mse_r16   = sum(r**2 for r in rmse_r_g16)   / T
print('  G16 filter MSE: GPU = %.6f   R = %.6f' % (total_mse_gpu16, total_mse_r16))

print()
print('  Note: MSE measures in-sample fit quality (lower = better).')
print('  GPU and R use different RNGs, so exact equality is not expected.')
print('  Similar MSE means both filters converged to equivalent quality.')
print()

# ─── write CSV summary ────────────────────────────────────────────────────────
out_csv = os.path.join(out_dir, 'G16_gibbs', 'mse_summary.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time_step', 'rmse_gpu_g17', 'rmse_r_g17', 'rmse_gpu_g16', 'rmse_r_g16'])
    for i, t in enumerate(range(1, T+1)):
        w.writerow([t, '%.8f' % rmse_gpu_g17[i], '%.8f' % rmse_r_g17[i],
                       '%.8f' % rmse_gpu_g16[i], '%.8f' % rmse_r_g16[i]])
print('  Saved: %s' % out_csv)
