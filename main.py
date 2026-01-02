"""
Synthetic Data Generation for Alzheimer's Disease Research
using WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty)

This code generates synthetic gene expression data for Alzheimer's Disease research with integrated wPCA quality assessment
using advanced GAN techniques to maintain statistical properties of original data.
"""

import os
import re
import warnings
import pickle
import collections
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.stats import ks_2samp
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Import existing modules
from synthetic_data_generator import (
    WGANGP, validate_and_normalize_columns, dynamic_preprocessing,
    inverse_processing, create_wgan_gp_generator_discriminator,
    generate_wgan_gp_samples
)

# Import new quality assessment module
from data_quality_assessment import DataQualityAssessor


def generate_and_evaluate_with_wpca(dataset_name: str, run_number: int, 
                                   data_path: Path, num_samples: int = 100) -> Optional[Dict]:
    """Generate synthetic data and perform wPCA quality assessment"""
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name} - Run {run_number}")
    print(f"{'='*50}")
    
    try:
        # Create output directories
        output_dir = data_path / f"WGAN-GP_{run_number}"
        quality_dir = output_dir / "quality_assessment"
        output_dir.mkdir(exist_ok=True, parents=True)
        quality_dir.mkdir(exist_ok=True)
        
        # Load and process data (existing code)
        train_file = data_path / f"{dataset_name}_train.xlsx"
        train_df = pd.read_excel(train_file, index_col=0)
        train_df = validate_and_normalize_columns(train_df)
        
        gene_names = train_df.index.tolist()
        
        # Generate synthetic data (existing WGAN-GP code)
        # ... [existing generation code] ...
        
        # Save synthetic data
        synthetic_path = output_dir / f"{dataset_name}_synthetic_all.xlsx"
        synthetic_df.to_excel(synthetic_path)
        
        # Perform wPCA quality assessment
        print("\nPerforming wPCA quality assessment...")
        assessor = DataQualityAssessor(random_seed=42)
        
        # Prepare real data file
        real_temp_path = output_dir / f"{dataset_name}_real_temp.csv"
        train_df.to_csv(real_temp_path)
        
        # Run comprehensive quality assessment
        quality_report, _ = assessor.comprehensive_data_quality_assessment(
            real_temp_path, synthetic_path, quality_dir
        )
        
        if quality_report is not None:
            # Save quality report
            quality_report['dataset'] = dataset_name
            quality_report['run'] = run_number
            
            report_df = pd.DataFrame([quality_report])
            report_path = quality_dir / "quality_report.csv"
            report_df.to_csv(report_path, index=False)
            
            print(f"wPCA score: {quality_report.get('wpca_score', 'N/A'):.4f}")
            print(f"Overall quality score: {quality_report.get('overall_quality_score', 'N/A'):.4f}")
        
        # Clean up temporary file
        if real_temp_path.exists():
            real_temp_path.unlink()
        
        return {
            'dataset': dataset_name,
            'run': run_number,
            'wpca_score': quality_report.get('wpca_score', np.nan),
            'overall_quality': quality_report.get('overall_quality_score', np.nan),
            'output_dir': str(output_dir)
        }
        
    except Exception as e:
        print(f"Error in generation with wPCA assessment: {e}")
        return None

def main_with_wpca():
    """Main function with integrated wPCA assessment"""
    data_path = Path(r"b:/20230315-manuscript/AD/STD")
    
    # Find datasets
    base_names = set()
    for f in os.listdir(data_path):
        if f.endswith("_train.xlsx"):
            dataset_name = f.split("_train.xlsx")[0]
            test_file = data_path / f"{dataset_name}_test.xlsx"
            if test_file.exists():
                base_names.add(dataset_name)
                print(f"Found dataset: {dataset_name}")
    
    print(f"Found {len(base_names)} datasets: {list(base_names)}")
    
    # Store all results
    all_results = []
    
    # Process each dataset
    for dataset_idx, dataset_name in enumerate(base_names, 1):
        print(f"\n{'='*60}")
        print(f"Processing dataset [{dataset_idx}/{len(base_names)}]: {dataset_name}")
        print(f"{'='*60}")
        
        # Multiple runs per dataset
        for run in range(1, 11):
            result = generate_and_evaluate_with_wpca(
                dataset_name, run, data_path, num_samples=100
            )
            
            if result:
                all_results.append(result)
            else:
                print(f"Warning: Run {run} for dataset {dataset_name} failed")
    
    # Generate summary report
    if all_results:
        print(f"\n{'='*60}")
        print("All processing completed! Summary results:")
        print(f"{'='*60}")
        
        summary_df = pd.DataFrame(all_results)
        
        # Calculate statistics per dataset
        dataset_summary = summary_df.groupby('dataset').agg({
            'wpca_score': ['mean', 'std', 'min', 'max'],
            'overall_quality': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Save summary results
        summary_path = data_path / "wpca_assessment_summary.xlsx"
        dataset_summary_path = data_path / "dataset_wpca_statistics.xlsx"
        
        summary_df.to_excel(summary_path, index=False)
        dataset_summary.to_excel(dataset_summary_path)
        
        print("Summary results saved to:")
        print(f"- {summary_path}")
        print(f"- {dataset_summary_path}")
        
        # Print results
        print("\nwPCA scores per dataset:")
        for dataset, row in dataset_summary.iterrows():
            wpca_mean = row[('wpca_score', 'mean')]
            quality_mean = row[('overall_quality', 'mean')]
            print(f"  {dataset}: wPCA={wpca_mean:.4f}, Overall={quality_mean:.4f}")
    
    print(f"\nAll quality assessments saved in WGAN-GP_*/quality_assessment folders")


def run_batch_wpca_assessment():
    """Run batch wPCA assessment on existing generated data"""
    data_path = Path(r"b:/20230315-manuscript/AD/STD")
    real_data_dir = data_path / "original"
    output_base = data_path / "wpca_results"
    
    # Initialize assessor
    assessor = DataQualityAssessor(random_seed=42)
    
    # Run assessment for each WGAN-GP run
    for run in range(1, 11):
        synthetic_dir = data_path / f"WGAN-GP_{run}"
        if not synthetic_dir.exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"Running wPCA assessment for WGAN-GP Run {run}")
        print(f"{'='*60}")
        
        output_dir = output_base / f"run_{run}"
        summary = assessor.batch_quality_assessment(
            real_data_dir, synthetic_dir, output_dir
        )
        
        if summary is not None:
            print(f"Completed assessment for run {run}")
            print(f"Average wPCA score: {summary['wpca_score'].mean():.4f}")
            print(f"Average overall quality: {summary['overall_quality_score'].mean():.4f}")
        


# ============================================================================
# PyTorch GAN Components (Legacy Support)
# ============================================================================

class Generator(nn.Module):
    """PyTorch Generator for traditional GAN"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super(Generator, self).__init__()
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """PyTorch Discriminator for traditional GAN"""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(Discriminator, self).__init__()
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# TensorFlow WGAN-GP Components
# ============================================================================

class WGANGP(tf.keras.Model):
    """WGAN-GP model for stable GAN training"""
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, 
                 latent_dim: int, gp_weight: float = 10.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    def compile(self, g_optimizer: tf.keras.optimizers.Optimizer, 
                d_optimizer: tf.keras.optimizers.Optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size: int, real_samples: tf.Tensor, 
                         fake_samples: tf.Tensor) -> tf.Tensor:
        """Calculate gradient penalty for WGAN-GP"""
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = (alpha * real_samples) + ((1 - alpha) * fake_samples)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        gradients = tape.gradient(pred, [interpolated])[0]
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gp = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        return gp

    def train_step(self, real_samples: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single training step for WGAN-GP"""
        batch_size = tf.shape(real_samples)[0]
        
        # Train discriminator
        with tf.GradientTape() as d_tape:
            # Generate fake samples
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            fake_samples = self.generator(random_latent_vectors, training=True)
            
            # Calculate discriminator loss
            real_logits = self.discriminator(real_samples, training=True)
            fake_logits = self.discriminator(fake_samples, training=True)
            
            d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            gp = self.gradient_penalty(batch_size, real_samples, fake_samples)
            d_loss = d_cost + gp * self.gp_weight
            
        # Apply discriminator gradients
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        d_gradients, _ = tf.clip_by_global_norm(d_gradients, 0.1)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as g_tape:
            # Generate fake samples
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            fake_samples = self.generator(random_latent_vectors, training=True)
            
            # Calculate generator loss
            gen_logits = self.discriminator(fake_samples, training=True)
            g_loss = -tf.reduce_mean(gen_logits)
            
        # Apply generator gradients
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        g_gradients, _ = tf.clip_by_global_norm(g_gradients, 0.1)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {m.name: m.result() for m in self.metrics}


# ============================================================================
# Data Processing Utilities
# ============================================================================

def validate_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize column names (must start with Control or AD)"""
    new_columns = []
    pattern = re.compile(r'^(Control|AD)([^a-zA-Z]|$)', re.IGNORECASE)
    
    for col in df.columns:
        col_str = str(col).strip()
        match = pattern.match(col_str)
        
        if match:
            prefix = match.group(1).capitalize()
            suffix = col_str[len(match.group(1)):].strip()
            
            if suffix == "":
                suffix = "0"
                
            suffix = re.sub(r'\W+', '_', suffix).strip('_')
            new_col = f"{prefix}_{suffix}"
            new_columns.append(new_col)
        else:
            raise ValueError(f"Invalid column format: {col_str}, must start with Control or AD")
    
    # Check for duplicate column names
    if len(new_columns) != len(set(new_columns)):
        duplicates = [item for item, count in collections.Counter(new_columns).items() if count > 1]
        raise ValueError(f"Duplicate column names after normalization: {duplicates}")
    
    df.columns = new_columns
    return df


def dynamic_preprocessing(data: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, tuple]:
    """Dynamic preprocessing with dimensionality reduction"""
    n_features, n_samples = data.shape  # [genes, samples]
    
    # Transpose for preprocessing (samples × genes)
    data_T = data.T
    
    # Robust scaling
    scaler = RobustScaler(quantile_range=(5, 95))
    scaled_data = scaler.fit_transform(data_T)
    
    # Handle NaN values
    if np.isnan(scaled_data).any():
        scaled_data = np.nan_to_num(scaled_data)
        print("Warning: NaN values detected, replaced with 0")
    
    # Automatic dimension adjustment
    valid_components = min(n_components, data_T.shape[0]-1, data_T.shape[1]-1)
    if valid_components < 1:
        raise ValueError(f"Insufficient feature dimensions: {valid_components}")
    
    # Randomized SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=valid_components, 
                      algorithm='randomized',
                      random_state=42)
    
    reduced_data = svd.fit_transform(scaled_data)
    return reduced_data, (svd, scaler, data.shape)


def inverse_processing(reduced_data: np.ndarray, processor: tuple) -> np.ndarray:
    """Inverse transform processed data back to original space"""
    svd, scaler, original_shape = processor
    n_genes, n_samples = original_shape
    
    # Inverse transform
    reconstructed = svd.inverse_transform(reduced_data)  # [new_samples, genes]
    
    # Inverse scaling
    reconstructed = scaler.inverse_transform(reconstructed)
    
    # Transpose back to original format (genes × samples)
    return reconstructed.T  # [genes, new_samples]


# ============================================================================
# Model Creation Functions
# ============================================================================

def create_wgan_gp_generator_discriminator(data_dim: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Create WGAN-GP generator and discriminator models"""
    # Generator
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_dim=100),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(data_dim, activation='tanh')
    ])
    
    # Discriminator
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_dim=data_dim),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation=None)  # No activation for WGAN
    ])
    
    return generator, discriminator


def train_pytorch_gan(data: np.ndarray, epochs: int = 1000, batch_size: int = 32, 
                      lr: float = 0.0002) -> Tuple[Optional[Generator], Optional[MinMaxScaler]]:
    """Train traditional GAN using PyTorch"""
    try:
        n_samples, n_features = data.shape
        print(f"Training data shape: {data.shape}")
        
        # Scale data to [-1, 1] range for Tanh activation
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)
        data_tensor = torch.FloatTensor(data_scaled)
        
        # Initialize models
        generator = Generator(n_features, n_features, hidden_dim=256)
        discriminator = Discriminator(n_features, hidden_dim=256)
        
        # Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            for i in range(0, n_samples, batch_size):
                batch_data = data_tensor[i:i+batch_size]
                batch_size_current = batch_data.shape[0]
                
                # Real and fake labels with label smoothing
                real_labels = torch.ones(batch_size_current, 1) * 0.9
                fake_labels = torch.zeros(batch_size_current, 1) + 0.1
                
                # Train discriminator
                optimizer_D.zero_grad()
                
                # Real data
                real_outputs = discriminator(batch_data)
                real_loss = criterion(real_outputs, real_labels)
                
                # Fake data
                noise = torch.randn(batch_size_current, n_features)
                fake_data = generator(noise)
                fake_outputs = discriminator(fake_data.detach())
                fake_loss = criterion(fake_outputs, fake_labels)
                
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                
                # Train generator
                optimizer_G.zero_grad()
                fake_outputs = discriminator(fake_data)
                g_loss = criterion(fake_outputs, real_labels)
                g_loss.backward()
                optimizer_G.step()
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        
        return generator, scaler
        
    except Exception as e:
        print(f"Error in train_pytorch_gan: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# Data Generation Functions
# ============================================================================

def generate_samples_pytorch(generator: Generator, scaler: MinMaxScaler, 
                           original_data: pd.DataFrame, sample_count: int = 50) -> Optional[pd.DataFrame]:
    """Generate samples using PyTorch GAN"""
    try:
        n_genes = original_data.shape[0]
        print(f"Generating {sample_count} samples with {n_genes} genes")
        
        # Generate all samples at once
        noise = torch.randn(sample_count, n_genes)
        with torch.no_grad():
            generated_data = generator(noise)
        
        # Transform back to original scale
        generated_data = generated_data.numpy()
        generated_data = scaler.inverse_transform(generated_data)
        
        # Create sample names
        sample_names = [f"Generated_Sample_{i+1}" for i in range(sample_count)]
        
        # Transpose to match original format
        generated_array = generated_data.T  # (genes, samples)
        
        gene_names = original_data.index.tolist()
        
        # Ensure dimensions match
        if len(gene_names) != generated_array.shape[0]:
            print(f"Warning: Dimension mismatch. Got {len(gene_names)} genes, but generated {generated_array.shape[0]} rows")
            if len(gene_names) < generated_array.shape[0]:
                generated_array = generated_array[:len(gene_names), :]
            else:
                gene_names = gene_names[:generated_array.shape[0]]
        
        # Create DataFrame
        df = pd.DataFrame(generated_array, index=gene_names, columns=sample_names)
        
        print(f"Generated data stats - Min: {df.values.min():.4f}, Max: {df.values.max():.4f}, "
              f"Mean: {df.values.mean():.4f}, Std: {df.values.std():.4f}")
        
        return df
        
    except Exception as e:
        print(f"Error in generate_samples_pytorch: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_wgan_gp_samples(generator: tf.keras.Model, processor: tuple, 
                            latent_dim: int, gene_names: List[str], 
                            num_samples: int = 100) -> pd.DataFrame:
    """Generate samples using WGAN-GP"""
    # Generate noise
    noise = tf.random.normal([num_samples, latent_dim])
    
    # Generate reduced dimension samples
    synthetic_reduced = generator(noise, training=False)
    
    # Inverse transform to original space
    synthetic_data = inverse_processing(synthetic_reduced.numpy(), processor)
    
    # Split samples into Control and AD
    num_control = num_samples // 2
    num_ad = num_samples - num_control
    
    # Create column names
    control_columns = [f"Control_synth_{i}" for i in range(num_control)]
    ad_columns = [f"AD_synth_{i}" for i in range(num_ad)]
    all_columns = control_columns + ad_columns
    
    # Create DataFrame
    synthetic_df = pd.DataFrame(
        synthetic_data,
        index=gene_names,
        columns=all_columns
    )
    
    return synthetic_df, control_columns, ad_columns


# ============================================================================
# Evaluation Functions
# ============================================================================

def perform_ks_test(original_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                   gene_names: List[str]) -> Tuple[pd.DataFrame, float]:
    """Perform Kolmogorov-Smirnov test between original and synthetic data"""
    ks_results = {}
    
    for i, gene_name in enumerate(gene_names):
        orig_values = original_data.iloc[i].values
        synth_values = synthetic_data.iloc[i].values
        
        # KS test
        ks_stat, ks_pvalue = ks_2samp(orig_values, synth_values)
        ks_results[gene_name] = {
            'KS_Statistic': ks_stat,
            'KS_PValue': ks_pvalue
        }
    
    # Create results DataFrame
    results_df = pd.DataFrame(ks_results).T
    results_df.index.name = 'Gene'
    results_df = results_df.reset_index()
    
    # Calculate average KS statistic
    avg_ks = results_df['KS_Statistic'].mean()
    results_df['Average_KS_Statistic'] = avg_ks
    
    return results_df, avg_ks


def save_model_pytorch(generator: Generator, scaler: MinMaxScaler, 
                      model_dir: Path, file_prefix: str) -> bool:
    """Save PyTorch model and scaler"""
    try:
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Save generator model
        generator_path = model_dir / f"{file_prefix}_generator.pth"
        torch.save(generator.state_dict(), str(generator_path))
        
        # Save scaler
        scaler_path = model_dir / f"{file_prefix}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"Model saved to {generator_path}")
        print(f"Scaler saved to {scaler_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False


def load_model_pytorch(model_dir: Path, file_prefix: str, 
                      n_features: int) -> Tuple[Optional[Generator], Optional[MinMaxScaler]]:
    """Load PyTorch model and scaler"""
    try:
        generator_path = model_dir / f"{file_prefix}_generator.pth"
        scaler_path = model_dir / f"{file_prefix}_scaler.pkl"
        
        # Check if model files exist
        if not generator_path.exists() or not scaler_path.exists():
            return None, None
        
        # Load generator model
        generator = Generator(n_features, n_features, hidden_dim=256)
        generator.load_state_dict(torch.load(str(generator_path)))
        generator.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"Model loaded from {generator_path}")
        return generator, scaler
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_dataset_wgangp(dataset_name: str, run_number: int, data_path: Path, 
                          num_samples: int = 100) -> Optional[Dict[str, Any]]:
    """Process dataset using WGAN-GP"""
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name} - Run {run_number}")
    print(f"{'='*50}")
    
    try:
        # Create output directory
        output_dir = data_path / f"WGAN-GP_{run_number}"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        print(f"Loading dataset: {dataset_name}")
        train_file = data_path / f"{dataset_name}_train.xlsx"
        train_df = pd.read_excel(train_file, index_col=0)
        train_df = validate_and_normalize_columns(train_df)
        
        # Get gene names
        gene_names = train_df.index.tolist()
        
        # Dynamic preprocessing
        print("Performing data preprocessing...")
        X_train, processor = dynamic_preprocessing(train_df.values)
        
        # Set dimensions
        data_dim = X_train.shape[1]
        latent_dim = 100
        
        # Create WGAN-GP model
        generator, discriminator = create_wgan_gp_generator_discriminator(data_dim)
        wgan_gp = WGANGP(generator, discriminator, latent_dim)
        
        # Compile model
        wgan_gp.compile(
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9),
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        )
        
        # Prepare training data
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train.astype(np.float32))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)
        
        # Train model
        epochs = 50
        print(f"Starting WGAN-GP training...")
        
        for epoch in range(epochs):
            epoch_losses = []
            for batch in train_dataset:
                losses = wgan_gp.train_step(batch)
                epoch_losses.append(losses)
            
            if (epoch + 1) % 10 == 0:
                avg_d_loss = np.mean([l['d_loss'] for l in epoch_losses])
                avg_g_loss = np.mean([l['g_loss'] for l in epoch_losses])
                print(f"Epoch {epoch+1}/{epochs}, D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        # Generate samples
        print("Generating synthetic data...")
        synthetic_df, control_cols, ad_cols = generate_wgan_gp_samples(
            generator, processor, latent_dim, gene_names, num_samples
        )
        
        # Validate data integrity
        assert synthetic_df.shape == (len(gene_names), num_samples), \
            f"Data shape error: expected {(len(gene_names), num_samples)}, got {synthetic_df.shape}"
        
        # Save all data
        output_path = output_dir / f"{dataset_name}_synthetic_all.xlsx"
        synthetic_df.to_excel(output_path)
        print(f"Saved synthetic data to: {output_path}")
        
        # Save Control and AD separately
        synthetic_df[control_cols].to_excel(output_dir / f"{dataset_name}_synthetic_Control.xlsx")
        synthetic_df[ad_cols].to_excel(output_dir / f"{dataset_name}_synthetic_AD.xlsx")
        
        # Perform KS test
        print("Performing KS test...")
        ks_results_df, avg_ks = perform_ks_test(train_df, synthetic_df, gene_names)
        
        # Save evaluation results
        eval_path = output_dir / f"{dataset_name}_ks_evaluation.xlsx"
        ks_results_df.to_excel(eval_path, index=False)
        print(f"KS evaluation results saved to: {eval_path}")
        print(f"Average KS statistic: {avg_ks:.4f}")
        
        # Clear memory
        del generator, discriminator, wgan_gp
        tf.keras.backend.clear_session()
        
        print(f"Run {run_number} completed!")
        
        return {
            'dataset': dataset_name,
            'run': run_number,
            'avg_ks': avg_ks,
            'output_dir': str(output_dir)
        }
        
    except FileNotFoundError as e:
        print(f"Error: Data file not found - {str(e)}")
        return None
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_dataset_pytorch_gan(file_path: Path, output_dir: Path, epochs: int = 1000, 
                               sample_count: int = 50, use_existing_model: bool = True) -> bool:
    """Process dataset using PyTorch GAN"""
    try:
        print(f"\nProcessing file: {file_path.name}")
        
        # Load data
        df = pd.read_csv(file_path, index_col=0)
        print(f"Original data shape: {df.shape}")
        
        # Separate Control and AD data
        control_cols = [col for col in df.columns if any(x in str(col).lower() 
                       for x in ['control', 'ctrl'])]
        ad_cols = [col for col in df.columns if any(x in str(col).lower() 
                   for x in ['ad', 'alzheimer'])]
        
        control_data = df[control_cols]
        ad_data = df[ad_cols]
        
        print(f"Control data shape: {control_data.shape}")
        print(f"AD data shape: {ad_data.shape}")
        
        # Create model directory
        model_dir = output_dir / "models"
        file_prefix = file_path.stem
        
        # Process Control data
        print("\n=== Processing Control Data ===")
        control_data_t = control_data.T
        print(f"Transposed Control data shape: {control_data_t.shape}")
        
        generator_c, scaler_c = None, None
        if use_existing_model:
            generator_c, scaler_c = load_model_pytorch(
                model_dir, f"{file_prefix}_ctrl", control_data_t.shape[1]
            )
        
        if generator_c is None or scaler_c is None:
            print("Training new GAN model for Control data...")
            generator_c, scaler_c = train_pytorch_gan(
                control_data_t.values, epochs=epochs, batch_size=16
            )
            
            if generator_c is not None:
                save_model_pytorch(generator_c, scaler_c, model_dir, f"{file_prefix}_ctrl")
        
        # Generate Control samples
        if generator_c is not None and scaler_c is not None:
            print("Generating Control samples...")
            generated_control_df = generate_samples_pytorch(
                generator_c, scaler_c, control_data, sample_count
            )
            
            if generated_control_df is not None:
                output_file = output_dir / f"generated_control_{file_path.name}"
                generated_control_df.to_csv(output_file)
                print(f"Generated Control samples saved to {output_file}")
        
        # Process AD data
        print("\n=== Processing AD Data ===")
        ad_data_t = ad_data.T
        print(f"Transposed AD data shape: {ad_data_t.shape}")
        
        generator_a, scaler_a = None, None
        if use_existing_model:
            generator_a, scaler_a = load_model_pytorch(
                model_dir, f"{file_prefix}_AD", ad_data_t.shape[1]
            )
        
        if generator_a is None or scaler_a is None:
            print("Training new GAN model for AD data...")
            generator_a, scaler_a = train_pytorch_gan(
                ad_data_t.values, epochs=epochs, batch_size=16
            )
            
            if generator_a is not None:
                save_model_pytorch(generator_a, scaler_a, model_dir, f"{file_prefix}_AD")
        
        # Generate AD samples
        if generator_a is not None and scaler_a is not None:
            print("Generating AD samples...")
            generated_ad_df = generate_samples_pytorch(
                generator_a, scaler_a, ad_data, sample_count
            )
            
            if generated_ad_df is not None:
                output_file = output_dir / f"generated_AD_{file_path.name}"
                generated_ad_df.to_csv(output_file)
                print(f"Generated AD samples saved to {output_file}")
                return True
        
        return False
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main execution function"""
    # Define data paths
    data_path = Path(r"b:/20230315-manuscript/AD/STD")
    
    # Find all available datasets
    base_names = set()
    for f in os.listdir(data_path):
        if f.endswith("_train.xlsx"):
            dataset_name = f.split("_train.xlsx")[0]
            test_file = data_path / f"{dataset_name}_test.xlsx"
            if test_file.exists():
                base_names.add(dataset_name)
                print(f"Found dataset: {dataset_name}")
    
    print(f"Found {len(base_names)} datasets: {list(base_names)}")
    
    # Store all evaluation results
    all_evaluations = []
    
    # Process each dataset
    for dataset_idx, dataset_name in enumerate(base_names, 1):
        print(f"\n{'='*60}")
        print(f"Processing dataset [{dataset_idx}/{len(base_names)}]: {dataset_name}")
        print(f"{'='*60}")
        
        # Generate 10 runs for each dataset
        for run in range(1, 11):
            result = process_dataset_wgangp(dataset_name, run, data_path, num_samples=100)
            
            if result:
                all_evaluations.append(result)
            else:
                print(f"Warning: Run {run} for dataset {dataset_name} failed")
        
        print(f"\nDataset {dataset_name} processing completed!")
    
    # Summary of all evaluations
    if all_evaluations:
        print(f"\n{'='*60}")
        print("All processing completed! Summary results:")
        print(f"{'='*60}")
        
        # Create summary table
        summary_df = pd.DataFrame(all_evaluations)
        
        # Calculate average KS statistics per dataset
        dataset_summary = summary_df.groupby('dataset')['avg_ks'].agg(['mean', 'std', 'min', 'max'])
        
        # Save summary results
        summary_path = data_path / "all_runs_summary.xlsx"
        dataset_summary_path = data_path / "dataset_summary_statistics.xlsx"
        
        summary_df.to_excel(summary_path, index=False)
        dataset_summary.to_excel(dataset_summary_path)
        
        print("Summary results saved to:")
        print(f"- {summary_path}")
        print(f"- {dataset_summary_path}")
        
        # Print average KS statistics per dataset
        print("\nAverage KS statistics per dataset:")
        for dataset, row in dataset_summary.iterrows():
            print(f"  {dataset}: {row['mean']:.4f} (std: {row['std']:.4f})")
    
    print(f"\nAll generated data saved to {data_path} in folders WGAN-GP_1 to WGAN-GP_10")
    print("Each folder contains:")
    print("  - datasetname_synthetic_all.xlsx: All synthetic data")
    print("  - datasetname_synthetic_Control.xlsx: Control samples")
    print("  - datasetname_synthetic_AD.xlsx: AD samples")
    print("  - datasetname_ks_evaluation.xlsx: KS evaluation results")


# ============================================================================
# Legacy PyTorch GAN Execution (Optional)
# ============================================================================

def run_pytorch_gan_pipeline():
    """Run the PyTorch GAN pipeline for legacy CSV data"""
    input_directory = Path(r"B:\20230315-manuscript\AD\STD\original")
    base_output_directory = Path(r"B:\20230315-manuscript\AD\STD")
    
    num_runs = 10
    for i in range(num_runs):
        output_directory = base_output_directory / f"GAN_{i+1:02d}"
        output_directory.mkdir(exist_ok=True, parents=True)
        
        print(f"Starting run {i+1}/{num_runs}, output directory: {output_directory}")
        
        # Get all CSV files
        csv_files = list(input_directory.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in input directory")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        
        success_count = 0
        for csv_file in csv_files:
            if process_dataset_pytorch_gan(
                csv_file, output_directory, epochs=500, 
                sample_count=50, use_existing_model=(i > 0)
            ):
                success_count += 1
        
        print(f"\nCompleted run {i+1}: {success_count}/{len(csv_files)} files processed successfully")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Option 1: Generate new data with wPCA assessment
    # main_with_wpca()
    
    # Option 2: Run wPCA assessment on existing data
    run_batch_wpca_assessment()
    
    # Uncomment to run PyTorch GAN pipeline for legacy data
    # run_pytorch_gan_pipeline()
