"""
Quality Score evaluation: PCA + five distribution metrics.

Computes an overall quality score for every original/synthetic pair
and saves a consolidated CSV with all results.

Functions:
    load_and_prepare_data()
    pca_comparison()
    calculate_distribution_metrics()
    visualize_pca_comparison()
    comprehensive_data_quality_assessment()
    run_quality_assessment()     # public CLI entry point
"""

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from utils.plotting import save_boxplot

# Restrict BLAS/MKL threads for multiprocessing
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# ----------------------------------------------------------------------
# 1. Data loading & alignment (specific to this module)
# ----------------------------------------------------------------------
def load_and_prepare_data(real_path, synth_path):
    """
    Load two expression datasets, transpose them to `samples × genes`,
    and return aligned DataFrames with common genes.
    """
    # Read
    if real_path.endswith('.csv'):
        real_df = pd.read_csv(real_path, index_col=0)
    elif real_path.endswith(('.xlsx', '.xls')):
        real_df = pd.read_excel(real_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {real_path}")

    if synth_path.endswith('.csv'):
        synth_df = pd.read_csv(synth_path, index_col=0)
    elif synth_path.endswith(('.xlsx', '.xls')):
        synth_df = pd.read_excel(synth_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {synth_path}")

    # Transpose: rows = samples, columns = genes
    real_df = real_df.T
    synth_df = synth_df.T

    # Common genes
    common = real_df.columns.intersection(synth_df.columns)
    real_df = real_df[common]
    synth_df = synth_df[common]

    # Remove duplicate gene names
    real_df = real_df.loc[:, ~real_df.columns.duplicated()]
    synth_df = synth_df.loc[:, ~synth_df.columns.duplicated()]

    # Convert non‑numeric columns and drop all‑NaN
    for df in [real_df, synth_df]:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(axis=1, how='all', inplace=True)

    return real_df, synth_df


# ----------------------------------------------------------------------
# 2. PCA comparison (randomized SVD)
# ----------------------------------------------------------------------
def pca_comparison(real_data, synth_data, n_components=50):
    """
    Joint PCA on real + synthetic data. Returns a DataFrame with PC
    coordinates and the explained variance ratio array.
    """
    real_arr = np.nan_to_num(real_data.values)
    synth_arr = np.nan_to_num(synth_data.values)

    combined = np.vstack([real_arr, synth_arr])
    labels = np.array(['real'] * real_arr.shape[0] + ['synthetic'] * synth_arr.shape[0])

    # Z‑score
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    std[std == 0] = 1
    X = (combined - mean) / std

    max_comp = min(X.shape[0], X.shape[1])
    n_components = min(n_components, max_comp)

    U, S, Vt = randomized_svd(X, n_components=n_components, random_state=42)
    X_pca = U * S
    explained_variance = (S ** 2) / np.sum(S ** 2)

    df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    df['data_type'] = labels
    return df, explained_variance


# ----------------------------------------------------------------------
# 3. Distribution metrics
# ----------------------------------------------------------------------
def calculate_distribution_metrics(real_data, synth_data,
                                   full_correlation=False, corr_sample_size=3000):
    """
    Compute five metrics comparing the two distributions:
      - mean correlation
      - mean RMSE
      - variance correlation
      - gene‑gene correlation RMSE
      - PCA centre distance
    """
    real = real_data.values
    synth = synth_data.values

    metrics = {}

    # Mean vectors
    mean_real = real.mean(axis=0)
    mean_synth = synth.mean(axis=0)
    metrics['mean_correlation'] = np.corrcoef(mean_real, mean_synth)[0, 1]
    metrics['mean_rmse'] = np.sqrt(np.mean((mean_real - mean_synth) ** 2))

    # Variance vectors
    var_real = real.var(axis=0)
    var_synth = synth.var(axis=0)
    metrics['variance_correlation'] = np.corrcoef(var_real, var_synth)[0, 1]

    # Gene‑gene correlation RMSE
    n_genes = real.shape[1]
    if full_correlation:
        print(f"    Computing full gene-gene correlation ({n_genes}x{n_genes})...")
        corr_real = np.corrcoef(real.T)
        corr_synth = np.corrcoef(synth.T)
    else:
        sample_size = min(corr_sample_size, n_genes)
        if sample_size < n_genes:
            np.random.seed(42)
            idx = np.random.choice(n_genes, sample_size, replace=False)
            real_sub = real[:, idx]
            synth_sub = synth[:, idx]
        else:
            real_sub, synth_sub = real, synth
        corr_real = np.corrcoef(real_sub.T)
        corr_synth = np.corrcoef(synth_sub.T)
    metrics['correlation_rmse'] = np.sqrt(np.mean((corr_real - corr_synth) ** 2))

    # PCA centre distance
    n_pcs = min(50, real.shape[0], real.shape[1])
    U, S, Vt = randomized_svd(real, n_components=n_pcs, random_state=42)
    pca_real = U * S
    real_mean = np.mean(real, axis=0)
    # Project synthetic data onto the same PCs
    pca_synth = (synth - real_mean) @ Vt.T
    metrics['pca_center_distance'] = np.linalg.norm(pca_real.mean(axis=0) - pca_synth.mean(axis=0))

    return metrics


# ----------------------------------------------------------------------
# 4. PCA visualisation
# ----------------------------------------------------------------------
def visualize_pca_comparison(pca_df, explained_variance, output_path):
    """Save a 2×2 panel figure: scatter, density, cumulative variance, KS stats."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='data_type',
                    alpha=0.6, ax=axes[0, 0])
    axes[0, 0].set_title('PCA: PC1 vs PC2')
    sns.kdeplot(data=pca_df, x='PC1', hue='data_type', ax=axes[0, 1])
    axes[0, 1].set_title('PC1 Distribution')

    cum_var = np.cumsum(explained_variance)
    axes[1, 0].plot(range(1, len(cum_var) + 1), cum_var, 'b-')
    axes[1, 0].set_xlabel('Number of PCs')
    axes[1, 0].set_ylabel('Cumulative Explained Variance')

    n_show = min(5, pca_df.shape[1] - 1)
    ks_stats = []
    for i in range(n_show):
        pc = f'PC{i+1}'
        real_pc = pca_df[pca_df['data_type'] == 'real'][pc]
        syn_pc = pca_df[pca_df['data_type'] == 'synthetic'][pc]
        ks_stats.append(stats.ks_2samp(real_pc, syn_pc)[0])
    axes[1, 1].bar(range(1, n_show + 1), ks_stats)
    axes[1, 1].set_xlabel('Principal Component')
    axes[1, 1].set_ylabel('KS Statistic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return ks_stats


# ----------------------------------------------------------------------
# 5. Comprehensive quality assessment (single pair)
# ----------------------------------------------------------------------
def comprehensive_data_quality_assessment(real_path, synth_path, output_dir,
                                          weights=None, full_correlation=False):
    """
    Run the full pipeline for one original/synthetic pair.
    Returns a dictionary of metrics and the combined PCA DataFrame.
    """
    if weights is None:
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]   # mean_corr, mean_rmse, var_corr, corr_rmse, pca_dist

    print(f"  [1/4] Loading data...")
    real_data, synth_data = load_and_prepare_data(real_path, synth_path)

    print(f"  [2/4] Filtering zero-variance genes...")
    real_var = real_data.var(axis=0)
    synth_var = synth_data.var(axis=0)
    valid = (real_var > 0) & (synth_var > 0)
    real_data = real_data.loc[:, valid]
    synth_data = synth_data.loc[:, valid]
    print(f"       After filtering: {real_data.shape[1]} genes, "
          f"real samples {real_data.shape[0]}, synthetic samples {synth_data.shape[0]}")

    print(f"  [3/4] Running PCA...")
    t0 = time.time()
    pca_df, explained_var = pca_comparison(real_data, synth_data)
    print(f"       PCA finished in {time.time()-t0:.2f} sec")

    print(f"  [4/4] Calculating distribution metrics (full_correlation={full_correlation})...")
    t0 = time.time()
    dist_metrics = calculate_distribution_metrics(real_data, synth_data, full_correlation)
    print(f"       Metrics computed in {time.time()-t0:.2f} sec")

    ks_stats = visualize_pca_comparison(pca_df, explained_var,
                                        os.path.join(output_dir, 'pca_comparison.png'))

    # Build report
    report = {
        'n_real_samples': real_data.shape[0],
        'n_synthetic_samples': synth_data.shape[0],
        'n_common_genes': real_data.shape[1],
        'pca_explained_variance_ratio': explained_var.sum(),
        **dist_metrics,
        'avg_pc_ks_statistic': np.mean(ks_stats),
        'max_pc_ks_statistic': np.max(ks_stats),
    }

    # Penalties and raw score
    rmse_penalty = min(report['mean_rmse'] / 10, 1)
    corr_penalty = min(report['correlation_rmse'] / 0.5, 1)
    pca_penalty = min(report['pca_center_distance'] / 10, 1)

    raw_score = (weights[0] * report['mean_correlation'] +
                 weights[1] * (1 - rmse_penalty) +
                 weights[2] * report['variance_correlation'] +
                 weights[3] * (1 - corr_penalty) +
                 weights[4] * (1 - pca_penalty))
    report['overall_quality_score'] = max(0.0, min(1.0, raw_score))
    report['raw_score'] = raw_score
    report['rmse_penalty'] = rmse_penalty
    report['corr_penalty'] = corr_penalty
    report['pca_penalty'] = pca_penalty

    # Save per‑pair CSV and PCA coordinates
    pd.DataFrame([report]).to_csv(os.path.join(output_dir, 'data_quality_assessment.csv'), index=False)
    pca_df.to_csv(os.path.join(output_dir, 'pca_results.csv'), index=False)

    return report, pca_df


# ----------------------------------------------------------------------
# 6. Multiprocessing helpers
# ----------------------------------------------------------------------
def _init_worker():
    """Set environment variables for parallel workers."""
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def _process_one_task(args):
    """Unpack arguments and run quality assessment for one pair."""
    real_file, gen_file, model_type, run_name, out_dir, weights, full_correlation = args
    try:
        report, _ = comprehensive_data_quality_assessment(
            real_file, gen_file, out_dir, weights, full_correlation
        )
        report['real_file'] = os.path.basename(real_file)
        report['model_type'] = model_type
        report['run'] = run_name
        return report
    except Exception as e:
        print(f"Error in {model_type}/{run_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------
# 7. Public entry point for the CLI
# ----------------------------------------------------------------------
def run_quality_assessment(real_dir, synth_root, models, output_dir,
                           workers=None, weights=None, full_correlation=False,
                           task_timeout=900):
    """
    Run the five‑component quality assessment for every original/synthetic
    pair across multiple generative models.

    Parameters
    ----------
    real_dir : str
        Directory containing original CSV/XLSX files.
    synth_root : str
        Root directory with model sub‑folders (e.g., .../GAN, .../VAE).
    models : list of str
        Model names to evaluate (e.g., ['GAN', 'VAE']).
    output_dir : str
        Where to save the aggregated results and per‑pair plots.
    workers : int or None
        Number of parallel processes. Defaults to CPU count.
    weights : list of float or None
        Five weights for the quality score. Defaults to [0.3, 0.2, 0.2, 0.2, 0.1].
    full_correlation : bool
        If True, compute the full gene‑gene correlation matrix; otherwise use random subset.
    task_timeout : int
        Maximum seconds per task before it is considered failed.
    """
    # Default weights if not provided
    if weights is None:
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]

    # Collect all real files
    real_files = glob.glob(os.path.join(real_dir, "*.csv")) + \
                 glob.glob(os.path.join(real_dir, "*.xlsx"))
    if not real_files:
        raise FileNotFoundError(f"No real data files found in {real_dir}")

    # Suffix mapping for synthetic files
    suffix_map = {
        'GAN': '_all.csv',
        'WGAN-GP': '_synthetic_all.xlsx',
        'diffusion': '_all.csv',
        'VAE': '_generated_all.csv',
    }

    tasks = []
    for real_file in real_files:
        base_name = os.path.splitext(os.path.basename(real_file))[0]
        for model in models:
            model_dir = os.path.join(synth_root, model)
            if not os.path.isdir(model_dir):
                print(f"Warning: model directory {model_dir} not found, skipping.")
                continue
            run_folders = [d for d in glob.glob(os.path.join(model_dir, f"{model}_*"))
                           if os.path.isdir(d)]
            for run_folder in run_folders:
                run_name = os.path.basename(run_folder)
                suffix = suffix_map.get(model, '')
                gen_file = os.path.join(run_folder, f"{base_name}{suffix}")
                if not os.path.exists(gen_file):
                    continue
                out_dir = os.path.join(output_dir, model, run_name, base_name)
                os.makedirs(out_dir, exist_ok=True)
                tasks.append((real_file, gen_file, model, run_name, out_dir,
                              weights, full_correlation))

    if not tasks:
        raise RuntimeError("No matching original/synthetic pairs found. "
                           "Check directory structure and file suffixes.")

    print(f"Total tasks: {len(tasks)} | Workers: {workers or os.cpu_count()} | Timeout: {task_timeout}s")

    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_process_one_task, t): t for t in tasks}
        with tqdm(total=len(futures), desc="Quality assessment") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                model, run = task[2], task[3]
                try:
                    res = future.result(timeout=task_timeout)
                    if res:
                        results.append(res)
                    else:
                        failed.append((model, run, "returned None"))
                except TimeoutError:
                    print(f"\nTimeout: {model}/{run}")
                    failed.append((model, run, "timeout"))
                except Exception as e:
                    print(f"\nException: {model}/{run}: {e}")
                    failed.append((model, run, str(e)))
                pbar.update(1)

    if failed:
        pd.DataFrame(failed, columns=['model', 'run', 'reason']).to_csv(
            os.path.join(output_dir, 'failed_tasks.csv'), index=False)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, 'all_quality_assessments.csv'), index=False)

        # Global boxplot
        save_boxplot(df, x='model_type', y='overall_quality_score',
             output_path=os.path.join(output_dir, 'quality_score_by_model.png'),
             title='Overall Quality Score by Model',
             xlabel='Model', ylabel='Quality Score')
        print(f"Quality assessment finished. Results saved to {output_dir}")
        
        return df
    else:
        print("No valid results were collected.")
        return None
