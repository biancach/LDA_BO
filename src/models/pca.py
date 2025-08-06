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

        self.mask = mask.flatten() if mask is not None else None
        self.masked = mask is not None

    def train(self, x):
        self.original_shape = x.shape[1:]  # (C, H, W)
        batch_size, C, H, W = x.shape
        x_flat = x.reshape(batch_size, C, H * W)  # (B, C, H*W)

        if self.masked:
            x_flat = x_flat[:, :, self.mask]  # apply mask

        x_flat = x_flat.reshape(batch_size, -1)
        x_scaled = self.scaler.fit_transform(x_flat)
        self.pca.fit(x_scaled)

    def encode(self, x):
        batch_size, C, H, W = x.shape
        x_flat = x.reshape(batch_size, C, H * W)

        if self.masked:
            x_flat = x_flat[:, :, self.mask]

        x_flat = x_flat.reshape(batch_size, -1)
        x_scaled = self.scaler.transform(x_flat)
        z = self.pca.transform(x_scaled)
        return z

    def decode(self, z):
        batch_size = z.shape[0]
        C, H, W = self.original_shape

        decoded = self.pca.inverse_transform(z)
        decoded = self.scaler.inverse_transform(decoded)
        decoded = decoded.reshape(batch_size, C, -1)

        if self.masked:
            x_full = np.full((batch_size, C, H * W), np.nan, dtype=decoded.dtype)
            x_full[:, :, self.mask] = decoded
        else:
            x_full = decoded

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
        obj = cls(latent_dim=data['latent_dim'], mask=data['mask'])
        obj.pca = data['pca']
        obj.scaler = data['scaler']
        obj.original_shape = data['original_shape']
        return obj
