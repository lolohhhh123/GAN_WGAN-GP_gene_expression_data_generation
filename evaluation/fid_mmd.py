"""
MMD and Fréchet distance evaluation.
Compares original versus synthetic expression data across multiple models and runs.

Provides:
  - run_fid_mmd_evaluation() : entry point for the main CLI (evaluate --method FID_MMD)
  - Internal helpers: load_data_file, infer_labels_from_names, split_generated_by_duplicate_names,
    mmd_rbf, frechet_distance, evaluate_pair, process_one_pair,
    collect_original_files, build_gen_file_path, generate_summary_plots.

This module expects shared utilities in:
  - utils/loading.py  -> load_data_file, infer_labels_from_names, split_generated_by_duplicate_names
  - utils/metrics.py  -> mmd_rbf, frechet_distance

Called by: main.py evaluate --method FID_MMD [--real_dir ... --synth_root ... --models ... --output_dir ... --workers ...]
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.plotting import generate_model_metric_plots
import multiprocessing as mp

# ---------- import shared utilities (must exist in utils/) ----------
from utils.loading import (load_data_file,
                           infer_labels_from_names,
                           split_generated_by_duplicate_names)
from utils.metrics import mmd_rbf, frechet_distance
# ----------------------------------------------------------------------

# ---------- 1. Data pair evaluation ----------
def evaluate_pair(orig_data, orig_labels, gen_data, gen_labels,
                  standardize=True, n_pca=50):
    """
    Compute MMD^2 and Fréchet distance for a single original/generated pair.
    Handles PCA reduction, overall and per-group metrics.
    """
    # Remove constant features
    orig_std = np.std(orig_data, axis=0)
    non_const = orig_std > 0
    if not non_const.all():
        orig_data = orig_data[:, non_const]
        gen_data  = gen_data[:, non_const]

    # Standardize (fit on original)
    if standardize:
        scaler = StandardScaler()
        scaler.fit(orig_data)
        orig_data = scaler.transform(orig_data)
        gen_data  = scaler.transform(gen_data)

    # PCA
    n_comp = min(n_pca, orig_data.shape[0] - 1, orig_data.shape[1])
    if n_comp >= 2:
        pca = PCA(n_components=n_comp, random_state=42)
        orig_data = pca.fit_transform(orig_data)
        gen_data  = pca.transform(gen_data)

    # Overall metrics
    mmd_all = mmd_rbf(orig_data, gen_data)
    fid_all = np.nan
    if orig_data.shape[0] > orig_data.shape[1] and gen_data.shape[0] > gen_data.shape[1]:
        mu_o   = np.mean(orig_data, axis=0)
        sigma_o = np.cov(orig_data, rowvar=False)
        mu_g   = np.mean(gen_data, axis=0)
        sigma_g = np.cov(gen_data, rowvar=False)
        fid_all = frechet_distance(mu_o, sigma_o, mu_g, sigma_g)

    result = {'MMD_overall': mmd_all, 'FID_overall': fid_all}

    # Per-group (Control=0, AD=1)
    for group_val, group_name in [(0, 'Control'), (1, 'AD')]:
        mask_orig = (orig_labels == group_val)
        mask_gen  = (gen_labels == group_val)
        if not mask_orig.any() or not mask_gen.any():
            result[f'MMD_{group_name}'] = np.nan
            result[f'FID_{group_name}'] = np.nan
            continue

        orig_g = orig_data[mask_orig]
        gen_g  = gen_data[mask_gen]

        result[f'MMD_{group_name}'] = mmd_rbf(orig_g, gen_g)

        if orig_g.shape[0] > orig_g.shape[1] and gen_g.shape[0] > gen_g.shape[1]:
            mu_og   = np.mean(orig_g, axis=0)
            sigma_og = np.cov(orig_g, rowvar=False)
            mu_gg   = np.mean(gen_g, axis=0)
            sigma_gg = np.cov(gen_g, rowvar=False)
            result[f'FID_{group_name}'] = frechet_distance(mu_og, sigma_og, mu_gg, sigma_gg)
        else:
            result[f'FID_{group_name}'] = np.nan

    return result


# ---------- 2. Single task (for multiprocessing) ----------
def process_one_pair(orig_file, gen_file, model_name, run_id, gse_id, n_pca=50):
    """
    Load one original and one synthetic file, align genes, evaluate metrics.
    Returns a dict with identifiers and all metric values.
    """
    try:
        orig = load_data_file(orig_file)
        if orig['data'].shape[0] == 0:
            return None
        orig_labels = infer_labels_from_names(orig['sample_names'])

        gen = load_data_file(gen_file)
        if gen['data'].shape[0] == 0:
            return None
        gen_labels, _ = split_generated_by_duplicate_names(gen['sample_names'])
        if gen_labels is None:
            return None

        common_genes = np.intersect1d(orig['gene_names'], gen['gene_names'])
        if len(common_genes) == 0:
            return None

        o_idx = [np.where(orig['gene_names'] == g)[0][0] for g in common_genes]
        g_idx = [np.where(gen['gene_names'] == g)[0][0] for g in common_genes]
        orig_aligned = orig['data'][:, o_idx]
        gen_aligned  = gen['data'][:, g_idx]

        metrics = evaluate_pair(orig_aligned, orig_labels, gen_aligned, gen_labels,
                                standardize=True, n_pca=n_pca)
        return {'GSE': gse_id, 'Model': model_name, 'RunID': run_id, **metrics}
    except Exception as e:
        warnings.warn(f"Error processing {gse_id}/{model_name}/{run_id}: {e}")
        return None


# ---------- 3. File collection and path construction ----------
def collect_original_files(original_dir):
    """Return dict gse_id -> full path for all original CSV/Excel files."""
    files = glob.glob(os.path.join(original_dir, "*.csv")) + \
            glob.glob(os.path.join(original_dir, "*.xlsx"))
    orig = {}
    for f in files:
        base = os.path.basename(f)
        # Use first part before '_' as GSE ID (or whole name without extension)
        gse = base.split('_')[0] if '_' in base else os.path.splitext(base)[0]
        if gse in orig:
            warnings.warn(f"Duplicate GSE {gse}: {orig[gse]} and {f}")
        orig[gse] = f
    return orig


def build_gen_file_path(model_folder, model_prefix, orig_filename):
    """
    Given the run directory, model name, and original file name,
    return the full path to the expected synthetic file.
    """
    base = os.path.splitext(orig_filename)[0]
    suffix_map = {
        'GAN':       '_all.csv',
        'WGAN-GP':   '_synthetic_all.xlsx',
        'diffusion': '_all.csv',
        'VAE':       '_generated_all.csv',
    }
    suffix = suffix_map.get(model_prefix, '')  # fallback: use original filename
    if suffix:
        return os.path.join(model_folder, f"{base}{suffix}")
    else:
        return os.path.join(model_folder, orig_filename)


# ---------- 4. Summary plots ----------
def generate_summary_plots(results_df, output_dir):
    # Ensure numeric columns
    metric_cols = [c for c in results_df.columns if c.startswith('MMD_') or c.startswith('FID_')]
    for col in metric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    generate_model_metric_plots(results_df, output_dir, metric_cols, group_col='Model')

    # Text summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("EVALUATION SUMMARY\n\n")
        f.write(f"Comparisons per model:\n{results_df['Model'].value_counts().to_string()}\n\n")
        f.write("Average metrics per model:\n")
        f.write(results_df.groupby('Model')[['MMD_overall', 'FID_overall']].mean().to_string())


# ---------- 5. Main public function ----------
def run_fid_mmd_evaluation(real_dir, synth_root, models, output_dir, workers=None, n_pca=50):
    """
    Run MMD and Fréchet distance evaluation for multiple models.

    Parameters
    ----------
    real_dir : str
        Path to directory containing original data files (CSV/XLSX).
    synth_root : str
        Root directory where each model's run folders are stored.
        Expected structure:
            synth_root/
              GAN/
                GAN_01/
                  <dataset>_all.csv
                GAN_02/ ...
              WGAN-GP/
                WGAN-GP_01/ ...
              ...
    models : list of str
        Names of generative models to evaluate, e.g. ['GAN', 'WGAN-GP', 'diffusion', 'VAE'].
    output_dir : str
        Directory to save evaluation_results.csv and plots.
    workers : int or None
        Number of parallel workers. Defaults to cpu_count().
    n_pca : int
        Number of PCA components to retain (default 50).
    """
    # Collect original files
    orig_files = collect_original_files(real_dir)
    if not orig_files:
        raise FileNotFoundError(f"No original data files (CSV/XLSX) found in {real_dir}")

    # Build task list
    tasks = []
    for gse_id, orig_path in orig_files.items():
        orig_filename = os.path.basename(orig_path)
        for model_name in models:
            model_root = os.path.join(synth_root, model_name)
            if not os.path.isdir(model_root):
                print(f"Warning: model directory {model_root} not found, skipping {model_name}")
                continue
            # Find run folders matching <model_name>_*
            pattern = os.path.join(model_root, f"{model_name}_*")
            run_folders = [d for d in glob.glob(pattern) if os.path.isdir(d)]
            for run_folder in run_folders:
                run_id = os.path.basename(run_folder).split('_', 1)[1] if '_' in os.path.basename(run_folder) else '?'
                gen_file = build_gen_file_path(run_folder, model_name, orig_filename)
                if os.path.exists(gen_file):
                    tasks.append((orig_path, gen_file, model_name, run_id, gse_id))
                else:
                    print(f"  Missing synthetic file: {gen_file}")

    if not tasks:
        raise RuntimeError("No matching original/synthetic file pairs found. Check paths and file suffixes.")

    print(f"Total evaluation tasks: {len(tasks)}")
    print(f"Workers: {workers or os.cpu_count()}")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_pair, *t, n_pca): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                results.append(res)
            if i % 10 == 0:
                print(f"Completed {i}/{len(tasks)}")

    if not results:
        raise RuntimeError("No valid evaluation results were obtained.")

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    generate_summary_plots(df, output_dir)
    print(f"Evaluation finished. Results saved to {output_dir}")
