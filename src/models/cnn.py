# models/cnn.py
import torch
import torch.nn as nn
import torch.optim as optim

class CNN_AE(nn.Module):
    def __init__(self, latent_dim=20, activation='relu'):
        super(CNN_AE, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation function.")
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),  # (2, 64, 64) → (32, 64, 64)
            self.activation,
            nn.MaxPool2d(2, 2),  # → (32, 32, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # → (64, 32, 32)
            self.activation,
            nn.MaxPool2d(2, 2),  # → (64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # → (128, 16, 16)
            self.activation,
            nn.MaxPool2d(2, 2),  # → (128, 8, 8)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # → (256, 8, 8)
            self.activation,
            nn.MaxPool2d(2, 2)  # → (256, 4, 4)
        )
        
        self.latent_dim = latent_dim
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # → (256, 8, 8)
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            self.activation,

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # → (128, 16, 16)
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            self.activation,

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # → (64, 32, 32)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            self.activation,

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # → (32, 64, 64)
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)  # → (2, 64, 64)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def train_model(self, train_loader, val_loader=None, epochs=1000, lr=1e-4, device='cpu',
                    loss_fn=nn.MSELoss(), patience=10, save_path=None):
        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

        best_val_loss = float('inf')
        patience_counter = 0
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for batch in train_loader:
                batch_data = batch[0].to(device)
                optimizer.zero_grad()
                reconstructed, latent = self(batch_data)
                loss = loss_fn(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            training_losses.append(avg_train_loss)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_data = batch[0].to(device)
                        reconstructed, latent = self(batch_data)
                        loss = loss_fn(reconstructed, batch_data)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                validation_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if save_path:
                        torch.save(self.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break
            else:
                print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}")

        return training_losses, validation_losses
    
    def evaluate(self, x, device='cpu'):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            return self.forward(x)