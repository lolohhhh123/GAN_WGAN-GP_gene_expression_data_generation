"""
GAN data generation module.
Provides:
  - train_gan()          : train a GAN on a single class of samples.
  - generate_samples()   : create synthetic DataFrame with labelled sample names.
  - process_single_file(): orchestrate loading, training, generation, and merging for one file.
  - run_gan_pipeline()   : public entry point called by main.py (e.g., generate --model GAN).

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
import pickle
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------
#  Model architectures (unchanged from original stable version)
# ----------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------
def train_gan(data, epochs=1000, batch_size=32, lr=0.0002):
    """
    Train a GAN on a numpy array of shape (n_samples, n_features).
    Returns (generator, scaler). Returns (None, None) on failure.
    """
    try:
        n_samples, n_features = data.shape
        print(f"Training data shape: {data.shape}")

        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)
        data_tensor = torch.FloatTensor(data_scaled)

        generator = Generator(n_features, n_features, hidden_dim=256)
        discriminator = Discriminator(n_features, hidden_dim=256)

        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            for i in range(0, n_samples, batch_size):
                batch = data_tensor[i:i+batch_size]
                bs = batch.shape[0]

                real_labels = torch.ones(bs, 1) * 0.9
                fake_labels = torch.zeros(bs, 1) + 0.1

                # Discriminator
                optimizer_D.zero_grad()
                real_loss = criterion(discriminator(batch), real_labels)
                noise = torch.randn(bs, n_features)
                fake = generator(noise)
                fake_loss = criterion(discriminator(fake.detach()), fake_labels)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                # Generator
                optimizer_G.zero_grad()
                g_loss = criterion(discriminator(fake), real_labels)
                g_loss.backward()
                optimizer_G.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        return generator, scaler

    except Exception as e:
        print(f"Error in train_gan: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ----------------------------------------------------------------------
#  Sample generation
# ----------------------------------------------------------------------
def generate_samples(generator, scaler, gene_names, num_samples, label):
    """
    Generate a DataFrame with genes as rows and labelled samples as columns.
    label: 'Control' or 'AD' (used to name columns like Control_synth_1, AD_synth_1...)
    """
    n_genes = len(gene_names)
    noise = torch.randn(num_samples, n_genes)
    with torch.no_grad():
        generated = generator(noise).numpy()
    generated = scaler.inverse_transform(generated)

    sample_names = [f"{label}_synth_{i+1}" for i in range(num_samples)]
    df = pd.DataFrame(generated.T, index=gene_names, columns=sample_names)
    print(f"Generated {label} stats - Min: {df.values.min():.4f}, Max: {df.values.max():.4f}, "
          f"Mean: {df.values.mean():.4f}, Std: {df.values.std():.4f}")
    return df


# ----------------------------------------------------------------------
#  Model persistence (optional)
# ----------------------------------------------------------------------
def save_model(generator, scaler, model_dir, file_prefix):
    """Save generator state dict and scaler to disk."""
    os.makedirs(model_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(model_dir, f"{file_prefix}_generator.pth"))
    with open(os.path.join(model_dir, f"{file_prefix}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Model saved with prefix {file_prefix}")


def load_model(model_dir, file_prefix, n_features):
    """Load generator and scaler. Returns (generator, scaler) or (None, None)."""
    gen_path = os.path.join(model_dir, f"{file_prefix}_generator.pth")
    scl_path = os.path.join(model_dir, f"{file_prefix}_scaler.pkl")
    if not os.path.exists(gen_path) or not os.path.exists(scl_path):
        return None, None
    try:
        generator = Generator(n_features, n_features, hidden_dim=64)
        generator.load_state_dict(torch.load(gen_path))
        generator.eval()
        with open(scl_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded model from {gen_path}")
        return generator, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# ----------------------------------------------------------------------
#  Single dataset processing
# ----------------------------------------------------------------------
def process_single_file(file_path, output_dir, epochs=1000, sample_count_total=1000,
                        use_existing=False):
    """
    Train separate GANs for Control and AD samples, generate the requested number of
    samples, and save the merged *_all.csv file in output_dir.
    If use_existing is True, tries to load pre‑trained models from output_dir/models.
    """
    # ---- data loading ----
    from utils.loading import load_and_preprocess_data   # must return DataFrame (genes × samples)

    df = load_and_preprocess_data(file_path)
    if df is None:
        raise ValueError(f"Failed to load {file_path}")

    print(f"Original data shape: {df.shape}")
    gene_names = df.index.tolist()

    # ---- split columns into Control / AD based on name ----
    ctrl_cols = [c for c in df.columns if 'control' in c.lower() or 'ctrl' in c.lower()]
    ad_cols   = [c for c in df.columns if 'ad' in c.lower() or 'alzheimer' in c.lower()]
    if len(ctrl_cols) < 2 or len(ad_cols) < 2:
        raise ValueError(f"Not enough Control ({len(ctrl_cols)}) or AD ({len(ad_cols)}) samples.")

    ctrl_data = df[ctrl_cols]      # genes × samples
    ad_data   = df[ad_cols]

    print(f"Control samples: {ctrl_data.shape[1]}, AD samples: {ad_data.shape[1]}")

    model_dir = os.path.join(output_dir, "models")
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    num_ctrl_gen = sample_count_total // 2
    num_ad_gen   = sample_count_total - num_ctrl_gen

    # ---- Control model ----
    gen_c, sc_c = None, None
    ctrl_train = ctrl_data.T.values   # shape (n_samples, n_genes)
    if use_existing:
        gen_c, sc_c = load_model(model_dir, f"{base_name}_ctrl", ctrl_train.shape[1])

    if gen_c is None:
        print("Training Control GAN...")
        gen_c, sc_c = train_gan(ctrl_train, epochs=epochs, batch_size=16)
        if gen_c is None:
            raise RuntimeError("Control GAN training failed.")
        if use_existing:
            save_model(gen_c, sc_c, model_dir, f"{base_name}_ctrl")

    print(f"Generating {num_ctrl_gen} Control samples...")
    df_ctrl = generate_samples(gen_c, sc_c, gene_names, num_ctrl_gen, "Control")

    # ---- AD model ----
    gen_a, sc_a = None, None
    ad_train = ad_data.T.values
    if use_existing:
        gen_a, sc_a = load_model(model_dir, f"{base_name}_AD", ad_train.shape[1])

    if gen_a is None:
        print("Training AD GAN...")
        gen_a, sc_a = train_gan(ad_train, epochs=epochs, batch_size=16)
        if gen_a is None:
            raise RuntimeError("AD GAN training failed.")
        if use_existing:
            save_model(gen_a, sc_a, model_dir, f"{base_name}_AD")

    print(f"Generating {num_ad_gen} AD samples...")
    df_ad = generate_samples(gen_a, sc_a, gene_names, num_ad_gen, "AD")

    # ---- merge and save ----
    df_all = pd.concat([df_ctrl, df_ad], axis=1)
    out_path = os.path.join(output_dir, f"{base_name}_all.csv")
    df_all.to_csv(out_path)
    print(f"Merged synthetic file saved to {out_path}")

    # free GPU memory (if any)
    del gen_c, gen_a, sc_c, sc_a
    torch.cuda.empty_cache()

    return df_all


# ----------------------------------------------------------------------
#  Public pipeline – called by main.py
# ----------------------------------------------------------------------
def run_gan_pipeline(input_dir, output_dir, runs, samples_per_run, epochs):
    """
    Perform `runs` independent runs for each dataset in input_dir.
    Output goes into subfolders: output_dir/GAN_01, GAN_02, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.csv")) + \
            glob.glob(os.path.join(input_dir, "*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {input_dir}")

    print(f"Found {len(files)} dataset(s). Starting {runs} run(s) for each.")

    for run_idx in range(1, runs + 1):
        run_folder = os.path.join(output_dir, f"GAN_{run_idx:02d}")
        os.makedirs(run_folder, exist_ok=True)

        for fpath in files:
            try:
                process_single_file(fpath, run_folder,
                                    sample_count_total=samples_per_run,
                                    epochs=epochs,
                                    use_existing=False)
            except Exception as e:
                print(f"Warning: Run {run_idx}, file {os.path.basename(fpath)} failed: {e}")

    print("GAN pipeline finished.")