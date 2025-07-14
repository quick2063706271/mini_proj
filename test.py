import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================
# Step 1: Load and prepare data
# ==========================
# Assuming X is your HVG-filtered gene expression DataFrame (cells x genes)
# y is a vector of cell type labels (as integers)
# protein is a DataFrame of protein expression (cells x proteins)
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)  # assuming y is a Pandas Series
protein_tensor = torch.tensor(protein.values, dtype=torch.float32)

# Wrap in dataset
dataset = TensorDataset(X_tensor, y_tensor, protein_tensor)

# Split into training and validation sets (80/20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==========================
# Step 2: Set up GPU device
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================
# Step 3: Define Autoencoder with Classification and Protein Prediction Heads
# ==========================
class AutoencoderMultiTask(nn.Module):
    def __init__(self, input_dim=2000, latent_dim=64, num_classes=10, num_proteins=50):
        super(AutoencoderMultiTask, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.protein_head = nn.Linear(latent_dim, num_proteins)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        class_logits = self.classifier(z)
        protein_preds = self.protein_head(z)
        return recon, class_logits, protein_preds

# ==========================
# Step 4: Train Autoencoder with Multi-Task Learning and Validation
# ==========================
model = AutoencoderMultiTask(
    input_dim=X_tensor.shape[1], 
    latent_dim=64, 
    num_classes=len(np.unique(y)), 
    num_proteins=protein_tensor.shape[1]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
recon_loss_fn = nn.MSELoss()
class_loss_fn = nn.CrossEntropyLoss()
protein_loss_fn = nn.MSELoss()

# Optional loss weights
alpha, beta, gamma = 1.0, 1.0, 1.0

for epoch in range(10):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        x, y_batch, protein_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        optimizer.zero_grad()
        recon, logits, protein_preds = model(x)
        loss_recon = recon_loss_fn(recon, x)
        loss_class = class_loss_fn(logits, y_batch)
        loss_protein = protein_loss_fn(protein_preds, protein_batch)
        loss = alpha * loss_recon + beta * loss_class + gamma * loss_protein
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            x, y_batch, protein_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            recon, logits, protein_preds = model(x)
            loss_recon = recon_loss_fn(recon, x)
            loss_class = class_loss_fn(logits, y_batch)
            loss_protein = protein_loss_fn(protein_preds, protein_batch)
            loss = alpha * loss_recon + beta * loss_class + gamma * loss_protein
            total_val_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}")
