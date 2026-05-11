"""
Conditional VAE data generation module.
Provides:
  - train_vae()              : train Conditional VAE on labelled expression data.
  - generate_samples()       : produce synthetic DataFrame with Control_synth_... / AD_synth_... columns.
  - process_single_file()    : orchestrate loading, training, generation, and saving.
  - run_vae_pipeline()       : public entry point called by main.py (e.g., generate --model VAE).

Requires:
  - utils.loading.load_and_preprocess_data()   <- returns DataFrame (genes as index, samples as columns)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
#  Device and seed
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
#  Scalers (used internally for VAE normalization)
# ----------------------------------------------------------------------
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


# ----------------------------------------------------------------------
#  Model definition
# ----------------------------------------------------------------------
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim=1, latent_dim=32, hidden_dims=[128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim + condition_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ----------------------------------------------------------------------
#  Data loading for VAE (internal, as VAE needs normalization + label)
# ----------------------------------------------------------------------
def load_vae_data(file_path, normalize=True):
    """
    Load CSV/XLSX, transpose to (samples x genes), infer labels from sample names.
    Returns gene_names, data (float32), labels (float32, 0/1), scaler (if normalize).
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, index_col=0)
    else:
        raise ValueError("Unsupported file format. Use CSV or XLSX.")

    df = df.T  # rows = samples, columns = genes

    if df.isnull().any().any():
        df = df.dropna(axis=0, how='any')
        if df.empty:
            raise ValueError("All samples dropped due to NaN values.")

    sample_names = df.index.tolist()
    gene_names = df.columns.tolist()

    # Infer labels
    labels = []
    for s in sample_names:
        if 'control' in s.lower() or 'ctrl' in s.lower():
            labels.append(0)
        elif 'ad' in s.lower() or 'alzheimer' in s.lower():
            labels.append(1)
        else:
            # Fallback: raise error for safety
            raise ValueError(f"Sample name '{s}' does not contain Control/AD keyword.")
    labels = np.array(labels, dtype=np.float32).reshape(-1, 1)

    data = df.values.astype(np.float32)

    scaler = None
    if normalize:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        data = (data - mean) / std
        scaler = StandardScaler(mean.flatten(), std.flatten())

    return gene_names, data, labels, scaler


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------
def train_vae(model, train_loader, val_loader, epochs=200, lr=1e-3,
              patience=15, beta=1.0, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        for batch_x, batch_c in train_loader:
            batch_x, batch_c = batch_x.to(device), batch_c.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_c)
            loss, _, _ = loss_function(recon_x, batch_x, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_c in val_loader:
                batch_x, batch_c = batch_x.to(device), batch_c.to(device)
                recon_x, mu, logvar = model(batch_x, batch_c)
                loss, _, _ = loss_function(recon_x, batch_x, mu, logvar, beta)
                val_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        warnings.warn("No best model found, using last epoch model.")
    return model


# ----------------------------------------------------------------------
#  Sample generation
# ----------------------------------------------------------------------
def generate_samples(model, num_samples, label, gene_names, scaler=None, device='cpu'):
    """
    Generate `num_samples` for a given label (0=Control, 1=AD).
    Returns DataFrame with samples as columns (genes rows).
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        c = torch.full((num_samples, 1), label, dtype=torch.float32).to(device)
        generated = model.decode(z, c).cpu().numpy()

    if scaler is not None:
        generated = scaler.inverse_transform(generated)

    group_name = "Control" if label == 0 else "AD"
    sample_names = [f"{group_name}_synth_{i+1}" for i in range(num_samples)]
    df = pd.DataFrame(generated.T, index=gene_names, columns=sample_names)
    return df


# ----------------------------------------------------------------------
#  Single dataset processing
# ----------------------------------------------------------------------
def process_single_file(file_path, output_dir, epochs=200, batch_size=32,
                        latent_dim=32, hidden_dims=[128, 64],
                        beta=1.0, lr=1e-3, patience=15, val_split=0.2,
                        sample_count_total=1000, seed=42):
    """
    Train a Conditional VAE on one dataset, generate synthetic samples,
    and save the merged *_generated_all.csv file.
    """
    set_seed(seed)

    # Load and normalize data
    gene_names, data, labels, scaler = load_vae_data(file_path, normalize=True)
    n_samples, n_features = data.shape
    control_mask = (labels.flatten() == 0)
    ad_mask = (labels.flatten() == 1)
    n_ctrl_real = control_mask.sum()
    n_ad_real = ad_mask.sum()
    print(f"Loaded {n_samples} samples ({n_ctrl_real} Control, {n_ad_real} AD), {n_features} genes.")

    # Convert to tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(data_tensor, labels_tensor)

    # Train/validation split
    val_size = max(1, int(n_samples * val_split))
    train_size = n_samples - val_size
    if train_size < 1:
        raise ValueError(f"Training set too small ({train_size}). Reduce val_split.")
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    model = ConditionalVAE(input_dim=n_features, condition_dim=1,
                           latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)

    print("Training Conditional VAE...")
    model = train_vae(model, train_loader, val_loader,
                      epochs=epochs, lr=lr, patience=patience, beta=beta, device=device)

    # Generate samples
    num_control = sample_count_total // 2
    num_ad = sample_count_total - num_control
    print(f"Generating {num_control} Control + {num_ad} AD samples...")
    df_ctrl = generate_samples(model, num_control, 0, gene_names, scaler, device)
    df_ad   = generate_samples(model, num_ad, 1, gene_names, scaler, device)

    # Merge and save
    df_all = pd.concat([df_ctrl, df_ad], axis=1)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_generated_all.csv")
    df_all.to_csv(out_path)
    print(f"Synthetic data saved to {out_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    return df_all


# ----------------------------------------------------------------------
#  Public pipeline – called by main.py
# ----------------------------------------------------------------------
def run_vae_pipeline(input_dir, output_dir, runs, samples_per_run, epochs):
    """
    Perform `runs` independent runs for each dataset in input_dir.
    Output stored in output_dir/VAE_01, VAE_02, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.csv")) + \
            glob.glob(os.path.join(input_dir, "*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {input_dir}")

    print(f"Found {len(files)} dataset(s). Starting {runs} run(s) for each.")

    for run_idx in range(1, runs + 1):
        run_folder = os.path.join(output_dir, f"VAE_{run_idx:02d}")
        os.makedirs(run_folder, exist_ok=True)

        for fpath in files:
            try:
                process_single_file(fpath, run_folder,
                                    epochs=epochs,
                                    sample_count_total=samples_per_run,
                                    seed=42 + run_idx)
            except Exception as e:
                print(f"Warning: Run {run_idx}, file {os.path.basename(fpath)} failed: {e}")

    print("VAE pipeline finished.")