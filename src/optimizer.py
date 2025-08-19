import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import InterpolatedUnivariateSpline

import torch
import GPy
import torch
import gpytorch

from src.utils import generate_trajectory

class Optimizer:
    def __init__(self, trajectories, dts, 
                 cost_function,
                 D, latent_distribution, 
                 grid_x, grid_y, scaler,
                 device,
                 acquisition_function='LCB', mask=None, 
                 seed=None, n_init=10, train_init=True):
        
        self.trajectories = trajectories
        self.dts = dts

        self.D = D
        self.latent_distribution = latent_distribution

        self.cost_function = cost_function

        self.grid_x = grid_x
        self.grid_y = grid_y
        self.scaler = scaler

        self.scaler_gp_x = StandardScaler()
        self.scaler_gp_y = StandardScaler()

        if acquisition_function not in ['LCB', 'EI', 'LCB-LW']:
            raise ValueError("Invalid acquisition function. Choose 'LCB', 'EI', or 'LCB-LW'.")
        else:
            self.acquisition_function = acquisition_function

        self.device = device

        if mask is None:
            self.mask = np.zeros_like(grid_x, dtype=bool)
        else:
            self.mask = mask

        
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(42)

        if n_init > 0:
            self.X, self.Y = self.initialize_data(n_init)
        else:
            self.X, self.Y = None, None
        
        self.train_init = train_init
        self.n_init = n_init
        self.max_cost = np.inf
        self.counter = 0
    
    def initialize_data(self, n_init=10):
        # X = self.latent_distribution.draw_samples(n_init,'lhs')
        X = self.latent_distribution.resample(n_init).T
        Y = np.zeros((n_init, 1))
        for i in range(n_init):
            pred = self.decode(X[i, :])
            traj_preds = self.simulate_trajectories(pred)
            Y[i] = self.cost_function(self.trajectories, traj_preds)
        X = self.scaler_gp_x.fit_transform(X)
        Y = self.scaler_gp_y.fit_transform(Y)
        return X, Y
    
    def decode(self, z):
        pred = self.D(torch.tensor(z).float().reshape(1, -1).to(self.device))
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        pred = pred[0, :, :, :]
        pred = self.scaler.inverse_transform(pred.reshape(-1, 2)).reshape(2, *self.grid_x.shape)
        u_pred, v_pred = pred[0, :, :], pred[1, :, :]
        u_pred[self.mask] = 0
        v_pred[self.mask] = 0
        return u_pred, v_pred
    
    def simulate_trajectories(self, pred):
        u_pred, v_pred = pred
        traj_preds = {}
        for traj_i in range(len(self.trajectories)):
            x0, y0 = self.trajectories[traj_i][0, :]
            dt_vals = self.dts[traj_i] if isinstance(self.dts[0], list) else self.dts
            traj_preds[traj_i] = generate_trajectory(u_pred, v_pred, self.grid_x, self.grid_y, x0, y0, dt_vals)
        return traj_preds
    
    def optimize(self, n_iterations=50):
        for _ in range(n_iterations):
            # z0 = self.latent_distribution.draw_samples(1, 'lhs').flatten()
            z0 =self.latent_distribution.resample(1).T.flatten()
            z0 = self.scaler_gp_x.transform(z0.reshape(1, -1))
            
            if hasattr(self, "acq_jacobian"):
                z_opt = minimize(self.acq_evaluate, z0, jac=self.acq_jacobian)['x']
            else:
                z_opt = minimize(self.acq_evaluate, z0)['x']

            self.X = np.vstack((self.X, z_opt))
            
            z_opt = self.scaler_gp_x.inverse_transform(z_opt.reshape(1, -1))
            pred = self.decode(z_opt.flatten())
            traj_preds = self.simulate_trajectories(pred)
            
            C_opt = self.cost_function(self.trajectories, traj_preds)
            self.Y = np.vstack((self.Y, self.scaler_gp_y.transform(np.array([[C_opt]]))))
            
            self.train_model(self.X, self.Y)  
            if C_opt < self.max_cost:
                self.max_cost = C_opt
                print(f'Iteration {self.counter}: {C_opt}')

            self.counter += 1
    
    def get_optimal(self):
        idx = np.argmin(self.Y)
        z_opt = self.scaler_gp_x.inverse_transform(self.X[idx, :].reshape(1, -1))
        pred = self.decode(z_opt.flatten())
        traj_opt = self.simulate_trajectories(pred)
        
        u_opt, v_opt = pred
        u_opt[self.mask] = np.nan
        v_opt[self.mask] = np.nan

        return z_opt, u_opt, v_opt, traj_opt
    
    def get_top_k(self, k=5):
        idx = np.argsort(self.Y.flatten())[:k]
        z_top = self.scaler_gp_x.inverse_transform(self.X[idx, :])
        pred_top = [self.decode(z.flatten()) for z in z_top]
        traj_top = [self.simulate_trajectories(pred) for pred in pred_top]
        u_top = [pred[0] for pred in pred_top]
        v_top = [pred[1] for pred in pred_top]
        for i in range(k):
            u_top[i][self.mask] = np.nan
            v_top[i][self.mask] = np.nan
        return z_top, u_top, v_top, traj_top
    
    def get_top_k_mean(self, k=5):
        idx = np.argsort(self.Y.flatten())[:k]
        z_top = self.scaler_gp_x.inverse_transform(self.X[idx, :])
        pred_top = [self.decode(z.flatten()) for z in z_top]
        u_top = [pred[0] for pred in pred_top]
        v_top = [pred[1] for pred in pred_top]
        for i in range(k):
            u_top[i][self.mask] = np.nan
            v_top[i][self.mask] = np.nan

        u_top = np.mean(u_top, axis=0)
        v_top = np.mean(v_top, axis=0)

        return u_top, v_top

    

class GPOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.train_init:
            # self.kernel = GPy.kern.RBF(input_dim=self.X.shape[1], ARD=True)
            self.kernel = GPy.kern.Matern32(input_dim=self.X.shape[1], ARD=True)
            self.gp = GPy.models.GPRegression(self.X, self.Y, self.kernel)
            self.gp.optimize_restarts(verbose=False, messages=False)

    def set_XY(self, X, Y):
        self.X = X
        self.Y = Y
        # self.kernel = GPy.kern.RBF(input_dim=self.X.shape[1], ARD=True)
        self.kernel = GPy.kern.Matern32(input_dim=self.X.shape[1], ARD=True)
        self.gp = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        self.gp.set_XY(X, Y)

    def train_model(self, X, Y):
        self.gp.set_XY(X, Y)
        self.gp.optimize_restarts(verbose=False, messages=False)

    def mean_prediction(self, x):
        x = np.atleast_2d(x)
        mu = self.gp.predict(x)[0].flatten()
        return mu

    def acq_evaluate(self, x):
        x = np.atleast_2d(x)
        mu, var = self.gp.predict_noiseless(x)
        std = np.sqrt(var)
        self.kappa = 2

        if self.acquisition_function == 'LCB':
            return mu - self.kappa * std
        
        elif self.acquisition_function == 'EI':
            y_best = np.min(self.Y)  
            z = (y_best - mu) / std
            return - ((y_best - mu) * norm.cdf(z) + std * norm.pdf(z))
        
        
    def acq_jacobian(self, x):
        x = np.atleast_2d(x)
        mu, var = self.gp.predict_noiseless(x)
        std = np.sqrt(var)

        mu_jac, var_jac = self.gp.predictive_gradients(x)
        mu_jac = mu_jac[:, :, 0]
        std_jac = var_jac / (2*std)

        if self.acquisition_function == 'LCB':
            return (mu_jac - self.kappa * std_jac).flatten()
        
        elif self.acquisition_function == 'EI':
            y_min = np.min(self.Y, axis=0)
            std_jac = var_jac / (2*std)
            z = (y_min - mu) / std
            pdf = norm.pdf(z)
            cdf = norm.cdf(z)
            return -((y_min - mu) * pdf * mu_jac + (pdf + z * cdf) * std_jac).flatten()
        


        
class DeepGPOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.train_init:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.gp = DeepGP(input_dim=self.X.shape[1], device=self.device)
            self.train_model(self.X, self.Y)

    def set_XY(self, X, Y):
        self.X = X
        self.Y = Y
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp = DeepGP(input_dim=self.X.shape[1], device=self.device)
        self.train_model(self.X, self.Y)

    def train_model(self, X, Y):
        self.gp.train_model(X, Y, self.likelihood)

    def mean_prediction(self, x):
        self.gp.eval()
        self.likelihood.eval()

        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.likelihood(self.gp(x_tensor))
            mu = pred.mean

        return mu.cpu().numpy().flatten()
    
    def acq_evaluate(self, x):
        x = np.atleast_2d(x)
        with torch.no_grad():
            mu, var = self.gp.predict(x, self.likelihood)
        std = np.sqrt(var)
        self.kappa = 2

        if self.acquisition_function == 'LCB':
            acq = mu - self.kappa * std
        
        elif self.acquisition_function == 'EI':            
            y_best = np.min(self.Y)  
            z = (y_best - mu) / std
            acq = -((y_best - mu) * norm.cdf(z) + std * norm.pdf(z))

        return acq.flatten()


class DeepGP(gpytorch.models.ApproximateGP):
    
    def __init__(self, input_dim, device, num_inducing=50):
        self.device = device
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing).to(device)
        inducing_points = torch.randn(num_inducing, input_dim, device=self.device)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )        
        super().__init__(variational_strategy)

        # Neural network feature extractor
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8)  
        ).to(self.device)

        # Deep Kernel (RBF applied to learned features)
        self.mean_module = gpytorch.means.ConstantMean().to(self.device)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(self.device)

        self.to(self.device)

    def forward(self, x):
        x_transformed = self.feature_extractor(x)
        mean_x = self.mean_module(x_transformed)
        covar_x = self.covar_module(x_transformed)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def train_model(self, X, Y, likelihood, num_epochs=50, lr=0.01):
        self.train()
        likelihood.train()

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device).squeeze(-1)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.VariationalELBO(likelihood, self, num_data=X.shape[0])

        for _ in range(num_epochs):
            optimizer.zero_grad()
            output = self(X)
            loss = -mll(output, Y)  # Negative ELBO
            loss.backward()
            optimizer.step()

        self.eval()
        likelihood.eval()

    def predict(self, X_new, likelihood):
        self.eval()
        likelihood.eval()
        
        X_new = torch.tensor(X_new, dtype=torch.float32, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(self(X_new))
        
        return pred.mean.cpu().numpy(), pred.variance.cpu().numpy()
