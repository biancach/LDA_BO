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
latent_dim = 50
acquisition_function = 'LCB'
deep = True
save_data = True
#####


if deep:
    tag = ''
else:
    tag = 'GP_'

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

nsamples = 1000
delta = nt // nsamples
tsteps = [(i+1)*delta - 1 for i in range(nsamples)]
nsamples = len(tsteps)

x0_idx = [18, 32, 65-18]
y0_idx = [18, 32, 65-18]

num_traj = 3

print('Generating trajectories...')
initial_conditions = {}
for traj_i in range(num_traj):
    x0 = grid_x[x0_idx[traj_i], y0_idx[traj_i]]
    y0 = grid_y[x0_idx[traj_i], y0_idx[traj_i]]
    initial_conditions[traj_i] = (x0, y0)

trajectories = {}

for tstep in tsteps:
    trajectories[tstep] = {}
    for traj_i in range(num_traj):
        x0, y0 = initial_conditions[traj_i]

        u_i = u[tstep, :, :]
        v_i = v[tstep, :, :]
        traj = generate_trajectory(u_i, v_i, grid_x, grid_y, x0, y0, dts)
        trajectories[tstep][traj_i] = traj

base_dir = f'{project_root}/results/kolmogorov/bo/constant_traj'

tsteps_copy = tsteps.copy()
for tstep in tsteps:
    tstep_dir = os.path.join(base_dir, str(tstep))
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

for tstep in tsteps:
    print(f"Preparing optimizer for tstep {tstep}...")
    if deep:
        optimizer =  DeepGPOptimizer(trajectories[tstep], dts, trajectory_cost_function, D, latent_inputs, grid_x, grid_y, nn_scaler, device, acquisition_function)
    else:
        optimizer = GPOptimizer(trajectories[tstep], dts, trajectory_cost_function, D, latent_inputs, grid_x, grid_y, nn_scaler, device, acquisition_function)


    try:
        optimizer.optimize(n_iterations=n_iter)
        print(f"tstep {tstep} optimization complete.")
    except Exception as e:
        print(f"tstep {tstep} optimization failed with error: {e}")
        

    if save_data:
        base_path = f'{project_root}/results/kolmogorov/bo/constant_traj/{tstep}'
        base_path = os.path.join(base_path, subfolder)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        np.save(os.path.join(base_path, 'X.npy'), optimizer.X)
        np.save(os.path.join(base_path, 'Y.npy'), optimizer.Y)
        joblib.dump(optimizer.scaler_gp_x, os.path.join(base_path, 'scaler_gp_x.joblib'))
        joblib.dump(optimizer.scaler_gp_y, os.path.join(base_path, 'scaler_gp_y.joblib'))
        print(f"Saved data for tstep {tstep} in {base_path}")
