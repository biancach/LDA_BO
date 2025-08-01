# models/mlp.py
import torch
import torch.nn as nn
import torch.optim as optim

class MLP_AE(nn.Module):
    def __init__(self, input_dim, latent_dim, mask=None):
        super(MLP_AE, self).__init__()

        self.mask = mask if mask is not None else slice(None) 
        self.original_shape = None
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, input_dim)
        )
    
    def encode(self, x):
        self.original_shape = x.shape[2:]
        x = x[:, :, self.mask].reshape(x.shape[0], -1)
        z = self.encoder(x)
        return z

    def decode(self, z):
        decoded = self.decoder(z)  # shape: (batch_size, 2 * num_valid_pixels)
        batch_size = decoded.shape[0]
        H, W = self.original_shape
        n_valid = self.mask.sum()

        # Split decoded output into u and v
        decoded = decoded.view(batch_size, 2, n_valid)  # (batch_size, 2, num_valid_pixels)

        x = torch.full((batch_size, 2, H * W), float('nan'), dtype=decoded.dtype, device=decoded.device)

        # Fill in only the valid locations (broadcasting across batch and channel)
        x[:, :, self.mask.flatten()] = decoded

        # Reshape to (batch_size, 2, H, W)
        return x.view(batch_size, 2, H, W)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def train_model(self, train_loader, val_loader=None, epochs=1000, lr=1e-4, device='cpu', 
                    loss_fn=nn.MSELoss(), patience=10, save_path=None):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
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
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
                
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
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Optionally, return loss histories
        return training_losses, validation_losses
    
    def evaluate(self, x, device='cpu'):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            return self.forward(x)
