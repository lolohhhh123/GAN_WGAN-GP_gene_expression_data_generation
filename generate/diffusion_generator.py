"""
Conditional Diffusion data generation module.
Provides:
  - train_diffusion_model()      : train a denoising diffusion model on labelled expression data.
  - generate_samples()           : produce synthetic DataFrame with Control_synth_... / AD_synth_... columns.
  - process_single_file()        : orchestrate loading, training, generation, and saving.
  - run_diffusion_pipeline()     : public entry point called by main.py (e.g., generate --model diffusion).

Requires:
  - utils.loading.load_and_preprocess_data()   <- returns DataFrame (genes as index, samples as columns)
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ----------------------------------------------------------------------
#  Diffusion model components (unchanged from original)
# ----------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    """Sinusoidal time-step embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class MLP(nn.Module):
    """Denoising network that conditions on time and class label."""
    def __init__(self, input_dim, hidden_dim=256, time_dim=128, num_classes=2):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.label_emb = nn.Embedding(num_classes, time_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t)
        y_emb = self.label_emb(y)
        h = torch.cat([x, t_emb, y_emb], dim=-1)
        return self.net(h)


class DiffusionModel:
    """DDPM conditioned on class labels."""
    def __init__(self, input_dim, num_classes=2, beta_start=1e-4, beta_end=0.02,
                 num_timesteps=1000, device=device):
        self.device = device
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.denoise_net = MLP(input_dim, num_classes=num_classes).to(device)
        self.optimizer = optim.Adam(self.denoise_net.parameters(), lr=1e-3)

    def train_step(self, x, y):
        batch_size = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(x)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        pred_noise = self.denoise_net(x_t, t, y)
        loss = nn.MSELoss()(pred_noise, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sample(self, num_samples, y, device='cuda'):
        """Generate samples from pure noise given label tensor y."""
        self.denoise_net.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, self.input_dim, device=device)
            for t in reversed(range(self.num_timesteps)):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                pred_noise = self.denoise_net(x, t_batch, y)

                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bars[t]
                beta_t = self.betas[t]

                noise = torch.randn_like(x) if t > 0 else 0
                x = 1 / torch.sqrt(alpha_t) * (
                    x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * pred_noise
                ) + torch.sqrt(beta_t) * noise
        self.denoise_net.train()
        return x


# ----------------------------------------------------------------------
#  Data preprocessing for diffusion (standardize each gene)
# ----------------------------------------------------------------------
def fit_scaler(data):
    """
    Fit mean/std scaler on data (n_samples, n_genes).
    Returns mean (1, n_genes) and std (1, n_genes) for later inverse transform.
    """
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return mean, std

def transform(data, mean, std):
    return (data - mean) / std

def inverse_transform(data, mean, std):
    return data * std + mean


# ----------------------------------------------------------------------
#  Training routine
# ----------------------------------------------------------------------
def train_diffusion_model(data, labels, epochs=500, batch_size=64,
                          patience=50, min_epochs=100):
    """
    Train a DiffusionModel on data (n_samples, n_genes) with class labels (0/1).
    Returns trained model, mean, std, and gene names are kept separately.
    Uses early stopping based on training loss.
    """
    mean, std = fit_scaler(data)
    data_norm = transform(data, mean, std)

    X = torch.tensor(data_norm, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = data.shape[1]
    model = DiffusionModel(input_dim=input_dim, num_classes=2, device=device)

    best_loss = float('inf')
    no_improve = 0

    print(f"Starting diffusion training (min {min_epochs} epochs, patience {patience})...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            loss = model.train_step(batch_x, batch_y)
            epoch_loss += loss
        avg_loss = epoch_loss / len(loader)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience and (epoch + 1) >= min_epochs:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Training finished. Best loss: {best_loss:.6f}")
    return model, mean, std


# ----------------------------------------------------------------------
#  Sample generation (with batching to handle large numbers)
# ----------------------------------------------------------------------
def generate_samples(model, mean, std, gene_names, num_control, num_ad,
                     batch_size=100):
    """
    Generate num_control Control samples and num_ad AD samples.
    Returns a DataFrame with genes as rows and labelled columns.
    """
    generated_parts = []
    labels_list = []

    for cls, n in [(0, num_control), (1, num_ad)]:
        generated_cls = []
        for start in range(0, n, batch_size):
            cur = min(batch_size, n - start)
            y = torch.full((cur,), cls, dtype=torch.long, device=device)
            samples_norm = model.sample(cur, y)
            samples_norm = samples_norm.cpu().numpy()  # (cur, n_genes)
            samples = inverse_transform(samples_norm, mean, std)
            generated_cls.append(samples)
        all_cls = np.vstack(generated_cls)  # (n, n_genes)
        generated_parts.append(all_cls)
        labels_list.extend([cls] * n)

    all_data = np.vstack(generated_parts).T  # (n_genes, total_samples)

    ctrl_cols = [f"Control_synth_{i+1}" for i in range(num_control)]
    ad_cols   = [f"AD_synth_{i+1}" for i in range(num_ad)]
    df = pd.DataFrame(all_data, index=gene_names, columns=ctrl_cols + ad_cols)
    return df


# ----------------------------------------------------------------------
#  Single dataset processing
# ----------------------------------------------------------------------
def process_single_file(file_path, output_dir, epochs=500, batch_size=64,
                        sample_count_total=1000):
    """
    Load one CSV/XLSX file, split into Control/AD, train a conditional diffusion model,
    generate samples, and save the merged *_all.csv file.
    """
    from utils.loading import load_and_preprocess_data  # returns DataFrame (genes × samples)

    df = load_and_preprocess_data(file_path)
    if df is None:
        raise ValueError(f"Could not load {file_path}")

    print(f"Original data shape: {df.shape}")
    gene_names = df.index.tolist()

    # Identify Control / AD columns by keyword
    ctrl_cols = [c for c in df.columns if 'control' in c.lower() or 'ctrl' in c.lower()]
    ad_cols   = [c for c in df.columns if 'ad' in c.lower() or 'alzheimer' in c.lower()]
    if len(ctrl_cols) < 2 or len(ad_cols) < 2:
        raise ValueError(f"Not enough Control ({len(ctrl_cols)}) or AD ({len(ad_cols)}) samples in {file_path}")

    # Build training matrix: (n_samples, n_genes)
    # Original df has genes as rows -> transpose to get samples as rows
    data_train = df[ctrl_cols + ad_cols].T.values  # (n_samples, n_genes)
    labels = np.array([0]*len(ctrl_cols) + [1]*len(ad_cols), dtype=np.int64)

    # Train model
    model, mean, std = train_diffusion_model(data_train, labels,
                                             epochs=epochs, batch_size=batch_size,
                                             patience=50, min_epochs=100)

    # Generate
    num_control = sample_count_total // 2
    num_ad = sample_count_total - num_control
    print(f"Generating {num_control} Control + {num_ad} AD samples...")
    df_synth = generate_samples(model, mean, std, gene_names, num_control, num_ad)

    # Save merged file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_all.csv")
    df_synth.to_csv(out_path)
    print(f"Synthetic data saved to {out_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    return df_synth


# ----------------------------------------------------------------------
#  Public pipeline – called by main.py
# ----------------------------------------------------------------------
def run_diffusion_pipeline(input_dir, output_dir, runs, samples_per_run, epochs):
    """
    Perform `runs` independent runs for each dataset in input_dir.
    Output stored in output_dir/diffusion_01, diffusion_02, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.csv")) + \
            glob.glob(os.path.join(input_dir, "*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {input_dir}")

    print(f"Found {len(files)} dataset(s). Starting {runs} run(s) for each.")

    for run_idx in range(1, runs + 1):
        run_folder = os.path.join(output_dir, f"diffusion_{run_idx:02d}")
        os.makedirs(run_folder, exist_ok=True)

        for fpath in files:
            try:
                process_single_file(fpath, run_folder,
                                    epochs=epochs,
                                    sample_count_total=samples_per_run)
            except Exception as e:
                print(f"Warning: Run {run_idx}, file {os.path.basename(fpath)} failed: {e}")

    print("Diffusion pipeline finished.")