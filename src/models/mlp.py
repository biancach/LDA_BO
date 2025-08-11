# models/mlp.py
import torch
import torch.nn as nn
import torch.optim as optim

class MLP_AE(nn.Module):
    def __init__(self, input_dim, latent_dim, mask=None):
        super(MLP_AE, self).__init__()

        if mask is not None and mask.ndim > 1:
            mask = mask.flatten()

        self.mask = mask  # Should now always be 1D or None
        self.masked = self.mask is not None
        self.original_shape = None

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        # x: (batch_size, 2, H, W)
        batch_size, C, H, W = x.shape
        self.original_shape = (H, W)

        x_flat = x.reshape(batch_size, C, H * W)  # (B, 2, H*W)
        if self.masked:
            x_flat = x_flat[:, :, self.mask]  # (B, 2, n_valid)
        
        x_flat = x_flat.reshape(batch_size, -1)  # (B, 2 * n_valid)
        z = self.encoder(x_flat)
        return z

    def decode(self, z):
        # z: (batch_size, latent_dim)
        decoded = self.decoder(z)  # (batch_size, 2 * n_valid)
        batch_size = decoded.shape[0]
        H, W = self.original_shape

        if self.masked:
            n_valid = self.mask.sum()
        else:
            n_valid = decoded.shape[1] // 2

        decoded = decoded.view(batch_size, 2, n_valid)  # (B, 2, n_valid)

        if self.masked:
            x_full = torch.full((batch_size, 2, H * W), float('nan'), dtype=decoded.dtype, device=decoded.device)
            x_full[:, :, self.mask] = decoded
        else:
            x_full = decoded

        return x_full.view(batch_size, 2, H, W)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


    def train_model(self, train_loader, val_loader=None, epochs=1000, lr=1e-3, device='cpu', 
                    loss_fn=nn.MSELoss(), patience=20, save_path=None, lr_patience=5, lr_factor=0.5):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        # Scheduler: reduce LR by lr_factor if val loss doesn't improve for lr_patience epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True
        )

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        training_losses = []
        validation_losses = []
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            
            for batch in train_loader:
                batch_data = batch[0].to(device)
                optimizer.zero_grad()
                recon, latent = self(batch_data)
                loss = loss_fn(recon, batch_data)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_data.size(0)
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            training_losses.append(avg_train_loss)
            
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_data = batch[0].to(device)
                        recon, latent = self(batch_data)
                        loss = loss_fn(recon, batch_data)
                        val_loss += loss.item() * batch_data.size(0)
                avg_val_loss = val_loss / len(val_loader.dataset)
                validation_losses.append(avg_val_loss)

                # Step the scheduler based on validation loss
                scheduler.step(avg_val_loss)

                print(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f} - "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    if save_path:
                        torch.save(self.state_dict(), save_path)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        return training_losses, validation_losses

    
    def evaluate(self, x, device='cpu'):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            return self.forward(x)
