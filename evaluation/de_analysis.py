"""
Differential Expression (DE) analysis evaluation.
Compares real vs. synthetic DE gene lists using Jaccard, top‑k overlap,
Spearman correlation of logFC, and a permutation‑based p‑value.

Provides:
  - run_de_analysis()   : public entry point for main.py (evaluate --method DE_analysis)
  - run_de_analysis()   : perform t‑test, FDR correction, and compute comparison metrics
  - compare_de_lists()  : calculate Jaccard, top‑k, Spearman, permutation p‑value
  - process_one_pair()  : worker for parallel processing

Assumes shared utilities in:
  - utils/loading.py    : load_data_file, infer_labels_from_names, split_generated_by_duplicate_names,
                          collect_original_files, build_gen_file_path
  (If not available, the module still contains its own copies of these helpers.)
"""

import os
import re
import glob
import warnings
import logging
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, spearmanr
from statsmodels.stats.multitest import fdrcorrection
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.plotting import generate_model_metric_plots
import matplotlib.pyplot as plt
import seaborn as sns

# ========== TEMPORARY: local copies of loading helpers (TODO: move to utils/loading.py) ==========
def load_data_file(filepath, transpose=True, na_threshold=0.5, fill_na=True):
    """Load CSV/Excel file and return dict with 'data', 'sample_names', 'gene_names'.
       This function should eventually be imported from utils.loading."""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, header=0)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, header=0)
    else:
        raise ValueError(f"Unsupported format: {filepath}")

    gene_names = df.iloc[:, 0].astype(str).values
    sample_names = df.columns[1:].values
    data_mat = df.iloc[:, 1:].values.astype(np.float64)

    # Remove entirely NA rows/cols
    na_rows = np.isnan(data_mat).all(axis=1)
    if na_rows.any():
        data_mat = data_mat[~na_rows]
        gene_names = gene_names[~na_rows]
    na_cols = np.isnan(data_mat).all(axis=0)
    if na_cols.any():
        data_mat = data_mat[:, ~na_cols]
        sample_names = sample_names[~na_cols]

    if transpose:
        data_mat = data_mat.T
        rownames = sample_names
        colnames = gene_names
    else:
        rownames = gene_names
        colnames = sample_names

    # Merge duplicate gene names by taking the mean
    if len(np.unique(colnames)) < len(colnames):
        df_agg = pd.DataFrame(data_mat, index=rownames, columns=colnames)
        df_agg = df_agg.T.groupby(level=0).mean().T
        data_mat = df_agg.values
        colnames = df_agg.columns.values

    # Handle missing values
    if np.any(np.isnan(data_mat)):
        na_prop = np.isnan(data_mat).mean(axis=0)
        keep = na_prop <= na_threshold
        if not keep.all():
            data_mat = data_mat[:, keep]
            colnames = colnames[keep]
        if fill_na and np.any(np.isnan(data_mat)):
            col_means = np.nanmean(data_mat, axis=0)
            for j in range(data_mat.shape[1]):
                col = data_mat[:, j]
                mask = np.isnan(col)
                if mask.any():
                    col[mask] = col_means[j]

    return {'data': data_mat, 'sample_names': rownames, 'gene_names': colnames}

def infer_labels_from_names(sample_names):
    """Return array of 0/1 based on 'Control'/'AD' prefix."""
    labels = np.full(len(sample_names), -1, dtype=int)
    for i, name in enumerate(sample_names):
        if re.match(r'^AD', name, re.IGNORECASE):
            labels[i] = 1
        elif re.match(r'^Control', name, re.IGNORECASE):
            labels[i] = 0
        else:
            raise ValueError(f"Cannot infer label from: {name}")
    if -1 in labels:
        raise ValueError("Some samples have unknown prefix")
    return labels

def split_generated_by_duplicate_names(sample_names):
    """Assume all Control or all AD come first. Returns labels array and split index."""
    control_idx = [i for i, n in enumerate(sample_names) if re.match(r'^Control', n, re.IGNORECASE)]
    ad_idx = [i for i, n in enumerate(sample_names) if re.match(r'^AD', n, re.IGNORECASE)]
    if not control_idx or not ad_idx:
        return None, None
    if control_idx[0] < ad_idx[0]:
        split_at = ad_idx[0]
        labels = [0] * split_at + [1] * (len(sample_names) - split_at)
    else:
        split_at = control_idx[0]
        labels = [1] * split_at + [0] * (len(sample_names) - split_at)
    return np.array(labels), split_at

def collect_original_files(original_dir):
    """Return dict {gse_id: filepath} for all CSV/Excel files in a directory."""
    files = glob.glob(os.path.join(original_dir, "*.csv")) + glob.glob(os.path.join(original_dir, "*.xlsx"))
    orig = {}
    for f in files:
        base = os.path.basename(f)
        gse = base.split('_')[0] if '_' in base else os.path.splitext(base)[0]
        if gse in orig:
            warnings.warn(f"Duplicate GSE {gse}: {orig[gse]} and {f}")
        orig[gse] = f
    return orig

def build_gen_file_path(model_folder, model_prefix, orig_filename):
    """Map model name and original filename to synthetic file path."""
    base = os.path.splitext(orig_filename)[0]
    suffix_map = {
        'GAN': '_all.csv',
        'WGAN-GP': '_synthetic_all.xlsx',
        'diffusion': '_all.csv',
        'VAE': '_generated_all.csv',
    }
    suffix = suffix_map.get(model_prefix, '')
    if suffix:
        return os.path.join(model_folder, f"{base}{suffix}")
    else:
        return os.path.join(model_folder, orig_filename)
# ========== End of temporary loading helpers ==========


# Configure error logging
logging.basicConfig(
    filename=os.path.join(os.getcwd(), 'de_errors.log'),  # can be overridden later
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_de_analysis(expr_df, labels, condition_order=('Control', 'AD')):
    """
    Perform two‑group independent t‑test for each gene and return results.
    labels can be a list/array of 0/1 or 'Control'/'AD'.
    """
    if len(labels) != expr_df.shape[0]:
        raise ValueError(f"Length of labels ({len(labels)}) != number of rows ({expr_df.shape[0]})")

    if isinstance(labels, pd.Series):
        labels = labels.values
    if isinstance(labels[0], (int, np.integer)):
        labels = np.array(['Control' if l == 0 else 'AD' for l in labels])
    else:
        labels = np.array(labels)

    mask_control = labels == condition_order[0]
    mask_case = labels == condition_order[1]
    group_control = expr_df.iloc[mask_control]
    group_case = expr_df.iloc[mask_case]

    n_control = group_control.shape[0]
    n_case = group_case.shape[0]

    if n_control < 2 or n_case < 2:
        warnings.warn(f"Sample size too small: Control={n_control}, Case={n_case}. DE results will be NaN.")
        return pd.DataFrame({
            'logFC': np.nan,
            'pvalue': np.nan,
            'adj_pvalue': np.nan,
            'neg_log10_adj_p': np.nan
        }, index=expr_df.columns)

    logFC = group_case.mean(axis=0) - group_control.mean(axis=0)
    _, pvals = ttest_ind(group_case.values, group_control.values, axis=0, equal_var=False)
    _, adj_pvals = fdrcorrection(pvals, alpha=0.05)

    results = pd.DataFrame({
        'logFC': logFC,
        'pvalue': pvals,
        'adj_pvalue': adj_pvals
    }, index=expr_df.columns)
    results['neg_log10_adj_p'] = -np.log10(results['adj_pvalue'] + 1e-10)
    results = results.sort_values('adj_pvalue')
    return results


def compare_de_lists(real_de, synth_de, real_expr_df, real_labels, top_k=200, n_permutations=100):
    """
    Compare DE gene lists between real and synthetic data.
    Returns: jaccard_index, top_k_overlap, spearman_rho_logfc, p_value_vs_random,
             n_real_sig_genes, n_synth_sig_genes.
    """
    real_sig = set(real_de[real_de['adj_pvalue'] < 0.05].index.tolist())
    synth_sig = set(synth_de[synth_de['adj_pvalue'] < 0.05].index.tolist())

    real_ranked = real_de.reindex(real_de['logFC'].abs().sort_values(ascending=False).index).index.tolist()
    synth_ranked = synth_de.reindex(synth_de['logFC'].abs().sort_values(ascending=False).index).index.tolist()

    # Jaccard index
    intersection = real_sig & synth_sig
    union = real_sig | synth_sig
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0

    # Top-k overlap
    top_k_overlap = len(set(real_ranked[:top_k]) & set(synth_ranked[:top_k])) / top_k

    # Spearman correlation of logFC across common genes
    common_genes = set(real_de.index) & set(synth_de.index)
    real_logfc = real_de.loc[list(common_genes), 'logFC']
    synth_logfc = synth_de.loc[list(common_genes), 'logFC']
    rho, _ = spearmanr(real_logfc, synth_logfc)

    # Permutation test against background Jaccard distribution
    background_jaccards = []
    labels_array = np.array(real_labels)
    if not isinstance(real_expr_df, pd.DataFrame):
        real_expr_df = pd.DataFrame(real_expr_df, columns=real_de.index)
    for _ in range(n_permutations):
        shuffled_labels = np.random.permutation(labels_array)
        shuffled_label_str = ['Control' if l == 0 else 'AD' for l in shuffled_labels]
        perm_de = run_de_analysis(real_expr_df, shuffled_label_str)
        perm_sig = set(perm_de[perm_de['adj_pvalue'] < 0.05].index.tolist())
        perm_jaccard = len(real_sig & perm_sig) / len(real_sig | perm_sig) if len(real_sig | perm_sig) > 0 else 0
        background_jaccards.append(perm_jaccard)
    p_value = np.mean(np.array(background_jaccards) >= jaccard)

    return {
        'jaccard_index': jaccard,
        f'top_{top_k}_overlap': top_k_overlap,
        'spearman_rho_logfc': rho,
        'p_value_vs_random': p_value,
        'n_real_sig_genes': len(real_sig),
        'n_synth_sig_genes': len(synth_sig)
    }


def process_one_pair(orig_file, gen_file, model_name, run_id, gse_id, top_k=200, n_permutations=100):
    """
    Worker function for parallel processing: loads, aligns, runs DE, and compares.
    Returns a dict of metrics or None if the pair cannot be processed.
    """
    try:
        orig = load_data_file(orig_file)
        if orig['data'].shape[0] == 0:
            return None
        orig_labels_int = infer_labels_from_names(orig['sample_names'])
        if np.sum(orig_labels_int == 0) < 2 or np.sum(orig_labels_int == 1) < 2:
            warnings.warn(f"{gse_id} original: insufficient group sizes, skipping.")
            return None

        gen = load_data_file(gen_file)
        if gen['data'].shape[0] == 0:
            return None
        gen_labels_int, _ = split_generated_by_duplicate_names(gen['sample_names'])
        if gen_labels_int is None:
            return None
        if np.sum(gen_labels_int == 0) < 2 or np.sum(gen_labels_int == 1) < 2:
            warnings.warn(f"{gse_id}/{model_name}/{run_id}: insufficient synthetic group sizes, skipping.")
            return None

        orig_df = pd.DataFrame(orig['data'], index=orig['sample_names'], columns=orig['gene_names'])
        gen_df = pd.DataFrame(gen['data'], index=gen['sample_names'], columns=gen['gene_names'])
        common_genes = np.intersect1d(orig['gene_names'], gen['gene_names'])
        if len(common_genes) == 0:
            return None

        orig_aligned = orig_df[common_genes]
        gen_aligned = gen_df[common_genes]

        orig_labels_str = ['Control' if l == 0 else 'AD' for l in orig_labels_int]
        gen_labels_str = ['Control' if l == 0 else 'AD' for l in gen_labels_int]

        real_de = run_de_analysis(orig_aligned, orig_labels_str)
        synth_de = run_de_analysis(gen_aligned, gen_labels_str)

        metrics = compare_de_lists(
            real_de, synth_de,
            real_expr_df=orig_aligned,
            real_labels=orig_labels_int,
            top_k=top_k,
            n_permutations=n_permutations
        )
        metrics.update({'GSE': gse_id, 'Model': model_name, 'RunID': run_id})
        return metrics

    except Exception as e:
        error_msg = f"Error processing {gse_id}/{model_name}/{run_id}\n" + traceback.format_exc()
        logging.error(error_msg)
        with open('de_errors.log', 'a') as f:
            f.write(error_msg + "\n" + "="*80 + "\n")
        warnings.warn(f"Error processing {gse_id}/{model_name}/{run_id}: {e}")
        return None


import traceback  # make sure it's imported

def generate_summary_plots(results_df, output_dir):
    metric_cols = ['jaccard_index', 'top_200_overlap', 'spearman_rho_logfc', 'p_value_vs_random']
    generate_model_metric_plots(results_df, output_dir, metric_cols, group_col='Model')


def run_de_analysis(real_dir, synth_root, models, output_dir, workers=None,
                    top_k=200, n_permutations=100):
    """
    Public entry point for DE evaluation.
    Compares every real dataset with every synthetic run of each given model.

    Parameters
    ----------
    real_dir : str
        Path to directory containing original CSV/XLSX files.
    synth_root : str
        Root directory with model sub‑folders (e.g., synth_root/GAN/GAN_01/).
    models : list of str
        Model names to evaluate (e.g., ['GAN','VAE','diffusion']).
    output_dir : str
        Directory to save the results CSV and plots.
    workers : int or None
        Number of parallel processes.
    top_k : int, default 200
        Top‑k overlap cutoff.
    n_permutations : int, default 100
        Number of permutations for the background p‑value.
    """
    orig_files = collect_original_files(real_dir)
    if not orig_files:
        raise FileNotFoundError(f"No original data files found in {real_dir}")

    tasks = []
    for gse_id, orig_file in orig_files.items():
        orig_filename = os.path.basename(orig_file)
        for model in models:
            model_root = os.path.join(synth_root, model)
            if not os.path.isdir(model_root):
                print(f"Warning: model directory {model_root} not found, skipping.")
                continue
            pattern = os.path.join(model_root, f"{model}_*")
            for mf in glob.glob(pattern):
                if not os.path.isdir(mf):
                    continue
                run_id = os.path.basename(mf).split('_', 1)[1] if '_' in os.path.basename(mf) else '?'
                gen_file = build_gen_file_path(mf, model, orig_filename)
                if os.path.exists(gen_file):
                    tasks.append((orig_file, gen_file, model, run_id, gse_id))
                else:
                    pass  # file missing

    if not tasks:
        raise RuntimeError("No matching original/synthetic file pairs found. Check file suffixes and directories.")

    print(f"Total DE tasks: {len(tasks)}")
    print(f"Workers: {workers or os.cpu_count()}")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_pair, *t, top_k, n_permutations): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res:
                results.append(res)
            if i % 10 == 0:
                print(f"Completed {i}/{len(tasks)}")

    if not results:
        raise RuntimeError("No valid DE comparison results obtained.")

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "de_comparison_results.csv"), index=False)
    generate_summary_plots(df, output_dir)
    print(f"DE analysis finished. Results saved to {output_dir}")
