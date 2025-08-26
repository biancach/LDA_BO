import numpy as np
import joblib
import torch
import os
import sys

# Adjust the path to your project root directory where 'models' folder is located
project_root = os.path.abspath('../../../')  # or the relative path to your root from the notebook folder

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.cnn import CNN_AE
from src.utils import *
from src.optimizer import GPOptimizer, DeepGPOptimizer

from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

####
tstep = 13833
latent_dim = 20
acquisition_function = 'LCB'
save_data = True
n_iter = 100
#####

base_path = f'{project_root}/results/fvcom/bo/constant_flow/{tstep}'
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


num_traj = 5
num_exp = 1000

print('Generating trajectories...')
np.random.seed(1)

trajectories, trajectories_m, idx0 = {}, {}, {}

for exp_i in range(num_exp):
    # Select well-separated initial points using a greedy farthest point sampling
    selected = [coords[np.random.randint(len(coords))]]
    for _ in range(num_traj - 1):
        min_dists = cdist(coords, np.array(selected)).min(axis=1)
        selected.append(coords[np.argmax(min_dists)])
    selected = np.array(selected)

    # Initialize dictionaries
    trajectories[exp_i], trajectories_m = {}, {}
    idx0[exp_i] = {'x0': [], 'y0': []}

    # Generate trajectories
    for traj_i, (x_idx, y_idx) in enumerate(selected):
        x0 = grid_x_m[y_idx, x_idx]
        y0 = grid_y_m[y_idx, x_idx]

        traj_m = generate_trajectory(u_fill[tstep], v_fill[tstep], grid_x_m, grid_y_m, x0, y0, dts)
        trajectories_m[traj_i] = traj_m

        traj = np.array(sp_proj('inverse', traj_m[:,0], traj_m[:,1], 'm')).T
        trajectories[exp_i][traj_i] = traj

        idx0[exp_i]['x0'].append(x_idx)
        idx0[exp_i]['y0'].append(y_idx)


print('Saving trajectories...')
os.makedirs(base_path, exist_ok=True)
exp_vals = list(np.arange(num_exp))

for exp_i in range(num_exp):
    exp_str = f'exp_x0_{"_".join(map(str, idx0[exp_i]["x0"]))}_y0_{"_".join(map(str, idx0[exp_i]["y0"]))}'

    exp_dir = os.path.join(base_path, exp_str)
    if os.path.exists(exp_dir):
        if os.path.exists(os.path.join(base_path, exp_str, subfolder)):
            exp_vals.remove(exp_i)
            trajectories.pop(exp_i)
            print(f"Skipping existing experiment: {exp_str}")
    else:
        os.makedirs(exp_dir)

        for traj_i in range(num_traj):
            traj = np.array(trajectories[exp_i][traj_i]).reshape(-1, trajectories[exp_i][traj_i][0].__len__())
            traj_path = os.path.join(exp_dir, f'traj_{traj_i}.npy')
            np.save(traj_path, traj)


print('Loading model...')
activation = 'relu'
nn = CNN_AE(latent_dim=latent_dim)
nn.load_state_dict(torch.load(f"{project_root}/src/models/checkpoints/fvcom/cnn_{latent_dim}_model.pth"))
nn.to(device)
D = nn.decode

data, nn_scaler = transform_data_for_AE(u,v)
latent_space = get_latent_space(nn, data)

latent_inputs = gaussian_kde(latent_space)


print('Preparing optimization...')

for exp_i in exp_vals:
    print(f"Preparing optimizer for experiment {exp_i}...")
    optimizer =  DeepGPOptimizer(trajectories[exp_i], dts, trajectory_cost_function, D, latent_inputs, grid_x, grid_y, nn_scaler, device, acquisition_function)
    
    try:
        optimizer.optimize(n_iterations=n_iter)
        print(f"Sample {exp_i} optimization complete.")
    except Exception as e:
        print(f"Sample {exp_i} optimization failed with error: {e}")
        

    if save_data:
        exp_str = f'exp_x0_{"_".join(map(str, idx0[exp_i]["x0"]))}_y0_{"_".join(map(str, idx0[exp_i]["y0"]))}'
        exp_path = os.path.join(base_path, exp_str, subfolder)
        os.makedirs(exp_path, exist_ok=True)

        np.save(os.path.join(exp_path, 'X.npy'), optimizer.X)
        np.save(os.path.join(exp_path, 'Y.npy'), optimizer.Y)
        joblib.dump(optimizer.scaler_gp_x, os.path.join(exp_path, 'scaler_gp_x.joblib'))
        joblib.dump(optimizer.scaler_gp_y, os.path.join(exp_path, 'scaler_gp_y.joblib'))
        print(f"Sample {exp_i} data saved to {exp_path}.")
