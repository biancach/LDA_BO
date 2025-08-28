import numpy as np
import joblib
import torch
import os
import sys

from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

# Adjust the path to your project root directory where 'models' folder is located
project_root = os.path.abspath('../../../')  # or the relative path to your root from the notebook folder

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.cnn import CNN_AE
from src.utils import *
from src.optimizer import GPOptimizer, DeepGPOptimizer

####
latent_dim = 30
acquisition_function = 'LCB'
save_data = True
n_iter = 100
seed = 0
#####


base_path = f'{project_root}/results/fvcom/bo/constant_traj'
subfolder = f'{acquisition_function}_{latent_dim}'

print('Loading data...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grid_x = np.load(f'{project_root}/data/fvcom/grid_x.npy')
grid_y = np.load(f'{project_root}/data/fvcom/grid_y.npy')
grid_x_m, grid_y_m = sp_proj('forward', grid_x, grid_y,  'm')

t = np.load(f'{project_root}/data/fvcom/t.npy')
u = np.load(f'{project_root}/data/fvcom/u_interp.npy')
v = np.load(f'{project_root}/data/fvcom/v_interp.npy')

mask = np.isnan(u[0,:,:])

nt, ngrid, _ = u.shape
print(u.shape, v.shape, grid_x.shape, grid_y.shape, t.shape)

u_fill, v_fill = u.copy(), v.copy()
u_fill[:, mask] = 0
v_fill[:, mask] = 0

dt = 3600
ndays = 0.75
T = ndays*24*60*60
dts = [dt for _ in range(int(T / dt))]

safe_mask = binary_erosion(~mask, structure=np.ones((12, 12)))  
yy, xx = np.where(safe_mask)
coords = np.array(list(zip(xx, yy)))

nsamples = 1000
delta = nt // nsamples
tsteps = [(i+1)*delta - 1 for i in range(nsamples)]
nsamples = len(tsteps)

num_traj = 5

print('Generating trajectories...')
np.random.seed(seed)

initial_conditions = [coords[np.random.randint(len(coords))]]
for _ in range(1, num_traj):
    min_dists = cdist(coords, np.array(initial_conditions)).min(axis=1)
    initial_conditions.append(coords[np.argmax(min_dists)])
initial_conditions = np.array(initial_conditions)

idx_str = f'x0_{"_".join(map(str, initial_conditions[:,0]))}_y0_{"_".join(map(str, initial_conditions[:,1]))}'
base_path = os.path.join(base_path, idx_str)


trajectories, trajectories_m = {}, {}
for tstep in tsteps:
    trajectories[tstep], trajectories_m[tstep] = {}, {}
    for traj_i in range(num_traj):
        x0_idx, y0_idx = initial_conditions[traj_i]
        x0, y0 = grid_x_m[y0_idx, x0_idx], grid_y_m[y0_idx, x0_idx]

        u_i = u_fill[tstep]
        v_i = v_fill[tstep]
        traj_m = generate_trajectory(u_fill[tstep], v_fill[tstep], grid_x_m, grid_y_m, x0, y0, dts)
        trajectories_m[tstep][traj_i] = traj_m
        
        traj = np.array(sp_proj('inverse', traj_m[:,0], traj_m[:,1], 'm')).T
        trajectories[tstep][traj_i] = traj


tsteps_copy = tsteps.copy()
for tstep in tsteps:
    tstep_dir = os.path.join(base_path, str(tstep))
    if os.path.exists(tstep_dir):
        if os.path.exists(os.path.join(tstep_dir, subfolder)):
            print(f"Data for tstep {tstep} already exists. Skipping...")
            tsteps_copy.remove(tstep)
    else:
        os.makedirs(tstep_dir)

        for traj_i in range(num_traj):
            traj = np.array(trajectories[tstep][traj_i]).reshape(-1, trajectories[tstep][traj_i][0].__len__())
            traj_path = os.path.join(tstep_dir, f'traj_{traj_i}.npy')
            np.save(traj_path, traj)

tsteps = tsteps_copy

print('Loading model...')
nn = CNN_AE(latent_dim=latent_dim)
nn.load_state_dict(torch.load(f"{project_root}/src/models/checkpoints/fvcom/cnn_{latent_dim}_model.pth"))
nn.to(device)
D = nn.decode

data, nn_scaler = transform_data_for_AE(u,v)
latent_space = get_latent_space(nn, data)

latent_inputs = gaussian_kde(latent_space)


print('Preparing optimization...')


for tstep in tsteps:
    print(f"Preparing optimizer for tstep {tstep}...")
    optimizer =  DeepGPOptimizer(trajectories[tstep], dts, trajectory_cost_function, D, latent_inputs, grid_x, grid_y, nn_scaler, device, acquisition_function)

    try:
        optimizer.optimize(n_iterations=n_iter)
        print(f"tstep {tstep} optimization complete.")
    except Exception as e:
        print(f"tstep {tstep} optimization failed with error: {e}")
        

    if save_data:
        tstep_dir = os.path.join(base_path, str(tstep))
        tstep_dir = os.path.join(tstep_dir, subfolder)
        if not os.path.exists(tstep_dir):
            os.makedirs(tstep_dir)

        np.save(os.path.join(tstep_dir, 'X.npy'), optimizer.X)
        np.save(os.path.join(tstep_dir, 'Y.npy'), optimizer.Y)
        joblib.dump(optimizer.scaler_gp_x, os.path.join(tstep_dir, 'scaler_gp_x.joblib'))
        joblib.dump(optimizer.scaler_gp_y, os.path.join(tstep_dir, 'scaler_gp_y.joblib'))
        print(f"Saved data for tstep {tstep} in {tstep_dir}")
