from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

class PCA_AE:
    def __init__(self, latent_dim, mask=None):
        self.n_components = latent_dim
        self.pca = PCA(n_components=latent_dim)
        self.scaler = StandardScaler()
        self.original_shape = None
        
        self.mask = mask if mask is not None else slice(None) 


    def train(self, x):
        self.original_shape = x.shape[1:]

        n_samples = x.shape[0]
        x_flat = x[:,:,self.mask].reshape(n_samples, -1)

        x_scaled = self.scaler.fit_transform(x_flat)
        self.pca.fit(x_scaled)

    def encode(self, x):
        batch_size, C, H, W = x.shape
        x = x.reshape(batch_size, C, H * W)  # (B, C, H*W)
        x = x[:, :, self.mask.flatten()]  # (B, C, n_valid_pixels)
        x = x.reshape(batch_size, -1)
        x_scaled = self.scaler.transform(x)
        z = self.pca.transform(x_scaled)
        return z

    def decode(self, z):
        batch_size = z.shape[0]
        C, H, W = self.original_shape

        decoded = self.pca.inverse_transform(z)
        decoded = self.scaler.inverse_transform(decoded)  # (B, C * n_valid)

        decoded = decoded.reshape(batch_size, C, -1)

        x_full = np.full((batch_size, C, H * W), np.nan, dtype=decoded.dtype)
        x_full[:, :, self.mask.flatten()] = decoded

        return x_full.reshape(batch_size, C, H, W)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def save(self, path):
        joblib.dump({
            'pca': self.pca,
            'scaler': self.scaler,
            'latent_dim': self.n_components,
            'original_shape': self.original_shape,
            'mask': self.mask
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls(latent_dim=data['latent_dim'])
        obj.pca = data['pca']
        obj.scaler = data['scaler']
        obj.original_shape = data['original_shape']
        obj.mask = data['mask']
        return obj
