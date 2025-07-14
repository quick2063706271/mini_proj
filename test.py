import torch.optim as optim

import torch.nn as nn

import torch
from torch.utils.data import TensorDataset, DataLoader

# Example: numpy array of shape (n_cells, n_genes)
X_tensor = torch.tensor(X.values, dtype=torch.float32)  # if X is a DataFrame

# Wrap in dataset
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)



class Autoencoder(nn.Module):
    def __init__(self, input_dim=2000, latent_dim=64):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out



model = Autoencoder(input_dim=2000, latent_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(10):  # adjust epochs
    model.train()
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
