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

####
tstep = 9407
latent_dim = 20
acquisition_function = 'LCB'
deep = False
save_data = True
#####

if deep:
    tag = ''
else:
    tag = 'GP_'

base_path = f'{project_root}/results/kolmogorov/bo/constant_flow/{tstep}'
subfolder = f'{tag}{acquisition_function}_{latent_dim}'

print('Loading data...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

L1, L2 = 2*np.pi, 2*np.pi
n1, n2 = 64, 64

grid_x = np.linspace(0,L1-L1/n1,n1)
grid_y = np.linspace(0,L2-L2/n2,n2)

grid_x, grid_y = np.meshgrid(grid_x, grid_y)

Kx = np.concatenate([np.arange(0, n1//2), np.arange(-n1//2, 0)]) * (2 * np.pi / L1)
Ky = np.concatenate([np.arange(0, n2//2), np.arange(-n2//2, 0)]) * (2 * np.pi / L2)

kx, ky = np.meshgrid(Kx,Ky)

dt_sim = 0.5
T_sim = 10000
nt = int(T_sim/dt_sim)

w = np.load(f'{project_root}/data/kolmogorov/vorticity.npy')
u = np.load(f'{project_root}/data/kolmogorov/u1.npy')
v = np.load(f'{project_root}/data/kolmogorov/u2.npy')

dt = 0.05
T = 1
dts = [dt for _ in range(int(T / dt))]

num_traj = 3
num_exp = 1000

print('Generating trajectories...')
np.random.seed(2)
trajectories = {}
idx0 = {}
for exp_i in range(num_exp):
    trajectories[exp_i] = {}
    idx0[exp_i] = {}
    idx0[exp_i]['x0'], idx0[exp_i]['y0'] = [], []
    for traj_i in range(num_traj):
        x0_idx = np.random.randint(4, n1-5)
        y0_idx = np.random.randint(4, n2-5)
        x0_i = grid_x[x0_idx, y0_idx]
        y0_i = grid_y[x0_idx, y0_idx]

        traj = generate_trajectory(u[tstep, :, :], v[tstep, :, :], grid_x, grid_y, x0_i, y0_i, dts)
        trajectories[exp_i][traj_i] = traj
        idx0[exp_i]['x0'].append(x0_idx)
        idx0[exp_i]['y0'].append(y0_idx)


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
nn = CNN_AE(latent_dim=latent_dim)
nn.load_state_dict(torch.load(f"{project_root}/src/models/checkpoints/kolmogorov/cnn_{latent_dim}_model.pth"))
nn.to(device)
D = nn.decode

data, nn_scaler = transform_data_for_AE(u,v)
latent_space = get_latent_space(nn, data)

latent_inputs = gaussian_kde(latent_space)


print('Preparing optimization...')

if latent_dim == 50:
    deep = True
    n_iter = 75
else:
    n_iter = 100

for exp_i in exp_vals:
    print(f"Preparing optimizer for experiment {exp_i}...")
    if deep:
        optimizer =  DeepGPOptimizer(trajectories[exp_i], dts, trajectory_cost_function, D, latent_inputs, grid_x, grid_y, nn_scaler, device, acquisition_function)
    else:
        optimizer = GPOptimizer(trajectories[exp_i], dts, trajectory_cost_function, D, latent_inputs, grid_x, grid_y, nn_scaler, device, acquisition_function)


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
