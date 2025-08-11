import os
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.interpolate import RegularGridInterpolator
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from pyproj import Proj, transform
import geopandas as gpd

project_root = os.path.abspath('../../../')  # or the relative path to your root from the notebook folder
states = gpd.read_file(f'{project_root}/data/massachusetts/s_08mr23.shp')
mass = states[states['NAME']=='Massachusetts']

# ============================ #
#        DATA TRANSFORMS      #
# ============================ #

def split_indices(n_samples, split=(0.4, 0.2, 0.4), seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    n_train = int(split[0] * n_samples)
    n_val = int(split[1] * n_samples)
    n_test = n_samples - n_train - n_val
    return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]

def transform_data_for_AE(u: np.ndarray, v: np.ndarray):
    """Scale and impute u, v field data for AE input."""
    data_raw = np.stack([u, v], axis=3)  # (N, H, W, 2)
    data_raw = np.transpose(data_raw, (0, 3, 1, 2))  # (N, 2, H, W)
    data_reshaped = data_raw.reshape(-1, 2)

    scaler = StandardScaler()
    valid_values = data_reshaped[~np.isnan(data_reshaped[:, 0])]
    scaled_values = scaler.fit_transform(valid_values)

    data_scaled = np.empty_like(data_reshaped)
    data_scaled[~np.isnan(data_reshaped[:, 0])] = scaled_values

    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_scaled)
    data_final = data_imputed.reshape(data_raw.shape)

    return data_final, scaler

def transform_data_for_AE_inverse(data, scaler: StandardScaler):
    """Inverse transform AE output to original scale."""
    data_flat = data.reshape(-1, 2)
    data_raw = scaler.inverse_transform(data_flat).reshape(data.shape)
    return data_raw


def sp_proj(proj_type, x, y, units):
    """
    proj_type : str
        Specifies the direction of the projection. 
        - 'forward', 'fwd', or 'f': Projects geographic coordinates (longitude, latitude) 
          to projected coordinates (x, y) in the specified units.
        - 'inverse', 'inv', or 'i': Projects from projected coordinates (x, y) 
          back to geographic coordinates (longitude, latitude).
    """
    
    if proj_type.lower() not in ['forward', 'fwd', 'f', 'inverse', 'inv', 'i']:
        raise ValueError("TYPE must be either 'forward' or 'inverse'")
    
    if np.shape(x) != np.shape(y):
        raise ValueError("x- and y- inputs must be the same size")
    
    if units.lower() not in ['meters', 'm', 'feet', 'survey feet', 'sf']:
        raise ValueError("Units must be either 'meters' or 'survey feet'")
    else:
        ur = 1.0 if units.lower() in ['meters', 'm'] else 0.3048  # Conversion factor to meters

    proj_params = {
        'proj': 'tmerc',
        'lat_0': 42.833333,
        'lon_0': -70.166667,
        'k': 0.999967,
        'x_0': 900000 * ur,
        'y_0': 0,
        'ellps': 'GRS80',
        'units': 'm' if units.lower() in ['meters', 'm'] else 'us-ft'
    }
    proj = Proj(proj_params)


    # Perform the projection
    if proj_type.lower() in ['forward', 'fwd', 'f']:
        xout, yout = proj(x, y)
    else:
        xout, yout = proj(x, y, inverse=True)

    return xout, yout


# ============================ #
#      MODEL UTILITIES        #
# ============================ #

def get_latent_space(model, data: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Extract latent representations from autoencoder model."""
    device = next(model.parameters()).device
    data_tensor = torch.tensor(data).float().to(device)

    model.eval()
    latent_list = []

    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size]
            _, latent = model(batch)
            latent_list.append(latent)

    return torch.cat(latent_list, dim=0).cpu().numpy().T

# ============================ #
#         PLOTTING            #
# ============================ #

def get_plot_limits(true, pred):
    """Return common vmin and vmax for plotting."""
    mask_true = ~np.isnan(true)
    mask_pred = ~np.isnan(pred)
    vmin = min(true[mask_true].min(), pred[mask_pred].min())
    vmax = max(true[mask_true].max(), pred[mask_pred].max())
    return vmin, vmax

def plot_field(ax, data, x, y, xmin, xmax, ymin, ymax, vmin=None, vmax=None, cmap='jet'):
    """Plot a 2D scalar field."""
    c = ax.pcolormesh(x, y, data, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.5)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect('equal')
    return c

def plot_trajectories(ax, data, x, y, trajectories, xmin, xmax, ymin, ymax, vmin=None, vmax=None, cmap='jet'):
    """Plot scalar field with overlaid trajectories."""
    c = plot_field(ax, data, x, y, xmin, xmax, ymin, ymax, vmin, vmax, cmap)
    
    for _, key in enumerate(trajectories):
        traj = trajectories[key]
        ax.plot(traj[0, 0], traj[0, 1], '*b', label='Start' if key == 0 else '')
        ax.plot(traj[-1, 0], traj[-1, 1], '*r', label='End' if key == 0 else '')
        ax.scatter(traj[:, 0], traj[:, 1], c='k', s=0.5)

    return c

def plot_trajectories_lines(trajectories, color='k'):
    for _, key in enumerate(trajectories):
        traj = trajectories[key]
        plt.scatter(traj[:, 0], traj[:, 1], c=color, s=0.5)
        plt.plot(traj[0, 0], traj[0, 1], '*b', label='Start' if key == 0 else '')


# ============================ #
#         TRAJECTORIES        #
# ============================ #

def generate_trajectory(u, v, x, y, x0, y0, dts):
    """Generate trajectory using bilinear interpolation."""
    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0

    u_interp = RegularGridInterpolator((y[:, 0], x[0, :]), u, method='cubic', bounds_error=False, fill_value=0)
    v_interp = RegularGridInterpolator((y[:, 0], x[0, :]), v, method='cubic', bounds_error=False, fill_value=0)

    current = np.array([x0, y0])
    trajectory = [current]

    for dt in dts:
        u_val = u_interp([current[1], current[0]])[0]
        v_val = v_interp([current[1], current[0]])[0]

        if np.isnan(u_val) or np.isnan(v_val):
            break

        current = current + dt * np.array([u_val, v_val])
        trajectory.append(current)

    return np.array(trajectory)

def trajectory_cost_function(true_trajs, pred_trajs):
    """Compute L2 trajectory error."""
    total_cost = 0
    n = len(true_trajs)

    for key in true_trajs:
        diff = np.linalg.norm(np.array(true_trajs[key]) - np.array(pred_trajs[key]), axis=1)
        total_cost += np.sum(diff)

    return total_cost / n


# ============================ #
#       ANALYSIS TOOLS        #
# ============================ #

def custom_KDE(data, weights=None, bw=None):
    """Compute FFT-based KDE."""
    data = data.flatten()

    if bw is None:
        try:
            kde = gaussian_kde(data, weights=weights)
            bw = np.sqrt(kde.covariance).flatten()
            bw = bw[0] if bw.size == 1 else 1
        except:
            bw = 1

    if bw < 1e-8:
        bw = 1

    return FFTKDE(bw=bw).fit(data, weights)


# ============================ #
#       FLUID MECHANICS        #
# ============================ #

def vort(u, v, kx, ky):
    """Compute vorticity in Fourier space."""
    fu = np.fft.fft2(u)
    fv = np.fft.fft2(v)
    fw = 1j * kx * fv - 1j * ky * fu
    return np.fft.ifft2(fw).real

def vort_FVCOM(u, v, x, y):
    """Finite-difference vorticity for FVCOM."""
    mask = np.isnan(u) | np.isnan(v)
    u = np.where(mask, np.nan, u)
    v = np.where(mask, np.nan, v)

    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]

    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    w = dv_dx - du_dy

    # threshold = 5 * np.nanstd(w)
    # w[np.abs(w) > threshold] = np.nan
    w[mask] = np.nan
    return w

def streamfunction_FVCOM(omega, x, y):
    """Solve Poisson equation for streamfunction from vorticity.
       Handles omega as 2D (ny, nx) or 3D (nt, ny, nx).
    """
    # Ensure omega is at least 3D: (nt, ny, nx)
    if omega.ndim == 2:
        omega = omega[None, :, :]  # add time axis
    
    nt, ny, nx = omega.shape
    mask = ~np.isnan(omega[0])  # same mask for all timesteps
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]

    # Map masked points to equation indices
    idx_map = -np.ones_like(mask, dtype=int)
    idx_map[mask] = np.arange(np.sum(mask))

    # Assemble sparse matrix A only once
    A = lil_matrix((np.sum(mask), np.sum(mask)))
    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            row = idx_map[j, i]
            A[row, row] = -2 / dx**2 - 2 / dy**2
            for di, dj, coeff in [(-1, 0, 1/dx**2), (1, 0, 1/dx**2),
                                  (0, -1, 1/dy**2), (0, 1, 1/dy**2)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and mask[nj, ni]:
                    A[row, idx_map[nj, ni]] = coeff
    A = A.tocsr()

    # Solve for each timestep
    psi_all = np.full_like(omega, np.nan)
    for t in range(nt):
        b = -omega[t][mask].flatten()
        psi_flat = spsolve(A, b)
        psi_t = np.full((ny, nx), np.nan)
        psi_t[mask] = psi_flat
        psi_all[t] = psi_t

    return psi_all if psi_all.shape[0] > 1 else psi_all[0]