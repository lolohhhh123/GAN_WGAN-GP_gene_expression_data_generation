"""
Conditional WGAN-GP data generation module.
Provides:
  - train_wgan_gp()          : train conditional WGAN-GP on labelled expression data.
  - generate_samples()       : produce synthetic DataFrame with Control_synth_... / AD_synth_... columns.
  - process_single_file()    : orchestrate loading, preprocessing, training, generation, and saving.
  - run_wgan_pipeline()      : public entry point called by main.py (e.g., generate --model WGAN-GP).

Requires:
  - utils.loading.load_and_preprocess_data()   <- returns DataFrame (genes as index, samples as columns)
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(42)
np.random.seed(42)


# ----------------------------------------------------------------------
#  Model components (unchanged from original stable version)
# ----------------------------------------------------------------------
class ConditionalWGANGP(tf.keras.Model):
    """Conditional WGAN with gradient penalty."""
    def __init__(self, generator, discriminator, latent_dim, cond_dim, gp_weight=50.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.gp_weight = gp_weight
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.d_steps = 0
        self.n_critic = 5

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_samples, real_conds, fake_samples, fake_conds):
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        cond_choice = tf.cast(tf.random.uniform([batch_size, 1]) > 0.5, tf.float32)
        interp_conds = cond_choice * real_conds + (1 - cond_choice) * fake_conds

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator([interpolated, interp_conds], training=True)

        gradients = tape.gradient(pred, [interpolated])[0]
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gp = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_samples, real_conds = data
        batch_size = tf.shape(real_samples)[0]
        random_conds = tf.random.uniform([batch_size, 1], 0, 1, dtype=tf.float32)
        random_latents = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as d_tape:
            fake_samples = self.generator([random_latents, random_conds], training=True)
            real_logits = self.discriminator([real_samples, real_conds], training=True)
            fake_logits = self.discriminator([fake_samples, random_conds], training=True)
            d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            gp = self.gradient_penalty(batch_size, real_samples, real_conds, fake_samples, random_conds)
            d_loss = d_cost + gp * self.gp_weight
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        self.d_steps += 1
        g_loss = tf.constant(0.0)
        if self.d_steps % self.n_critic == 0:
            with tf.GradientTape() as g_tape:
                fake_samples = self.generator([random_latents, random_conds], training=True)
                gen_logits = self.discriminator([fake_samples, random_conds], training=True)
                g_loss = -tf.reduce_mean(gen_logits)
            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}


# ----------------------------------------------------------------------
#  Preprocessing (WGAN-specific: RobustScaler + TruncatedSVD)
# ----------------------------------------------------------------------
def dynamic_preprocessing(data, n_components=5000):
    """
    Transpose (genes->samples), scale, and reduce dimensions.
    data: (n_genes, n_samples)
    Returns: reduced_data (n_samples, n_components), processor tuple (svd, scaler, original_shape)
    """
    n_features, n_samples = data.shape
    data_T = data.T  # (samples, genes)

    scaler = RobustScaler(quantile_range=(5, 95))
    scaled_data = scaler.fit_transform(data_T)
    if np.isnan(scaled_data).any():
        scaled_data = np.nan_to_num(scaled_data)

    valid_components = min(n_components, data_T.shape[0] - 1, data_T.shape[1] - 1)
    if valid_components < 1:
        raise ValueError(f"Not enough features for SVD: {valid_components}")

    svd = TruncatedSVD(n_components=valid_components, algorithm='randomized', random_state=42)
    reduced_data = svd.fit_transform(scaled_data)  # (samples, components)
    return reduced_data, (svd, scaler, data.shape)


def inverse_processing(reduced_data, processor):
    """Reverse SVD and scaling to recover original gene space (n_genes, n_samples)."""
    svd, scaler, original_shape = processor
    reconstructed = svd.inverse_transform(reduced_data)  # (samples, genes)
    reconstructed = scaler.inverse_transform(reconstructed)
    return reconstructed.T  # (genes, samples)


def create_conditional_generator_discriminator(latent_dim, cond_dim, output_dim):
    """Build Keras models for conditional generator and discriminator."""
    # Generator
    noise_input = tf.keras.Input(shape=(latent_dim,), name='noise')
    cond_input = tf.keras.Input(shape=(cond_dim,), name='condition')
    x = tf.keras.layers.Concatenate()([noise_input, cond_input])
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    out = tf.keras.layers.Dense(output_dim, activation=None)(x)
    generator = tf.keras.Model([noise_input, cond_input], out, name='conditional_generator')

    # Discriminator
    data_input = tf.keras.Input(shape=(output_dim,), name='data')
    cond_input_d = tf.keras.Input(shape=(cond_dim,), name='condition_d')
    y = tf.keras.layers.Concatenate()([data_input, cond_input_d])
    y = tf.keras.layers.Dense(1024, activation='relu')(y)
    y = tf.keras.layers.LeakyReLU(0.2)(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.Dense(512, activation='relu')(y)
    y = tf.keras.layers.LeakyReLU(0.2)(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.Dense(256, activation='relu')(y)
    y = tf.keras.layers.LeakyReLU(0.2)(y)
    out_d = tf.keras.layers.Dense(1, activation=None)(y)
    discriminator = tf.keras.Model([data_input, cond_input_d], out_d, name='conditional_discriminator')

    return generator, discriminator


# ----------------------------------------------------------------------
#  Sample generation
# ----------------------------------------------------------------------
def generate_samples(generator, processor, gene_names, num_control, num_ad, latent_dim=100):
    """
    Generate Control and AD samples using the conditional generator,
    then inverse process to gene space and return a DataFrame with labelled columns.
    """
    cond_dim = 1
    noise_c = tf.random.normal((num_control, latent_dim))
    cond_c = tf.zeros((num_control, cond_dim), dtype=tf.float32)
    synth_c = generator([noise_c, cond_c], training=False)

    noise_a = tf.random.normal((num_ad, latent_dim))
    cond_a = tf.ones((num_ad, cond_dim), dtype=tf.float32)
    synth_a = generator([noise_a, cond_a], training=False)

    all_reduced = tf.concat([synth_c, synth_a], axis=0).numpy()
    all_original = inverse_processing(all_reduced, processor)  # (genes, total_samples)

    ctrl_cols = [f"Control_synth_{i+1}" for i in range(num_control)]
    ad_cols   = [f"AD_synth_{i+1}" for i in range(num_ad)]
    df = pd.DataFrame(all_original, index=gene_names, columns=ctrl_cols + ad_cols)
    return df


# ----------------------------------------------------------------------
#  Training routine
# ----------------------------------------------------------------------
def train_wgan_gp(reduced_data, labels, latent_dim=100, cond_dim=1,
                  epochs=500, batch_size=32, min_epochs=100, patience=10):
    """
    Train a ConditionalWGANGP model on reduced_data (n_samples, n_features) and labels (0/1).
    Returns trained generator and discriminator.
    """
    output_dim = reduced_data.shape[1]
    generator, discriminator = create_conditional_generator_discriminator(latent_dim, cond_dim, output_dim)

    wgan = ConditionalWGANGP(generator, discriminator, latent_dim, cond_dim, gp_weight=50.0)
    wgan.compile(
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.9),
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005, beta_1=0.5, beta_2=0.9)
    )

    dataset = tf.data.Dataset.from_tensor_slices((reduced_data.astype(np.float32),
                                                  labels.astype(np.float32)))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    target_d_loss_range = (-5.0, 5.0)
    target_g_loss_range = (-10.0, 10.0)
    consecutive_good_epochs = 0

    print(f"Start training WGAN-GP (min {min_epochs} epochs)...")
    for epoch in range(epochs):
        epoch_losses = []
        for batch_data, batch_conds in dataset:
            losses = wgan.train_step((batch_data, batch_conds))
            epoch_losses.append(losses)
        avg_d = np.mean([l['d_loss'] for l in epoch_losses])
        avg_g = np.mean([l['g_loss'] for l in epoch_losses])

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, D Loss: {avg_d:.4f}, G Loss: {avg_g:.4f}")

        d_in = target_d_loss_range[0] <= avg_d <= target_d_loss_range[1]
        g_in = target_g_loss_range[0] <= avg_g <= target_g_loss_range[1]
        if d_in and g_in and (epoch + 1) >= min_epochs:
            consecutive_good_epochs += 1
            if consecutive_good_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            consecutive_good_epochs = 0

    return generator, discriminator


# ----------------------------------------------------------------------
#  Single dataset processing
# ----------------------------------------------------------------------
def process_single_file(file_path, output_dir, epochs=500, batch_size=32,
                        sample_count_total=1000):
    """
    Load a single CSV/XLSX file, split into Control/AD, train conditional WGAN,
    generate samples, and save the merged *_synthetic_all.xlsx file.
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

    # Build data matrix (genes × samples) and labels (0 for Control, 1 for AD)
    data_matrix = df[ctrl_cols + ad_cols].values  # genes × (control + ad)
    labels = np.array([0] * len(ctrl_cols) + [1] * len(ad_cols), dtype=np.float32).reshape(-1, 1)

    # Dimensionality reduction (WGAN-GP works in reduced space)
    reduced_data, processor = dynamic_preprocessing(data_matrix)

    # Train conditional model
    generator, discriminator = train_wgan_gp(reduced_data, labels, epochs=epochs, batch_size=batch_size)

    num_control = sample_count_total // 2
    num_ad = sample_count_total - num_control
    print(f"Generating {num_control} Control + {num_ad} AD samples...")
    df_synth = generate_samples(generator, processor, gene_names, num_control, num_ad)

    # Save merged file (xlsx, to keep consistency with original evaluation map)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_synthetic_all.xlsx")
    df_synth.to_excel(out_path)
    print(f"Synthetic data saved to {out_path}")

    # Cleanup
    del generator, discriminator
    tf.keras.backend.clear_session()
    return df_synth


# ----------------------------------------------------------------------
#  Public pipeline – called by main.py
# ----------------------------------------------------------------------
def run_wgan_pipeline(input_dir, output_dir, runs, samples_per_run, epochs):
    """
    Perform `runs` independent runs for each dataset in input_dir.
    Output stored in output_dir/WGAN-GP_01, WGAN-GP_02, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.csv")) + \
            glob.glob(os.path.join(input_dir, "*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {input_dir}")

    print(f"Found {len(files)} dataset(s). Starting {runs} run(s) for each.")

    for run_idx in range(1, runs + 1):
        run_folder = os.path.join(output_dir, f"WGAN-GP_{run_idx:02d}")
        os.makedirs(run_folder, exist_ok=True)

        for fpath in files:
            try:
                process_single_file(fpath, run_folder,
                                    epochs=epochs,
                                    sample_count_total=samples_per_run)
            except Exception as e:
                print(f"Warning: Run {run_idx}, file {os.path.basename(fpath)} failed: {e}")

    print("WGAN-GP pipeline finished.")