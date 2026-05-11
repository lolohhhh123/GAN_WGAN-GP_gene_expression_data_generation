"""
Merge quality assessment and evaluation results.

Merges the output of `quality_score.py` (all_quality_assessments.csv) with
`fid_mmd.py` (evaluation_results.csv) on GSE, model_type and run number.
Optionally filters to keep only GSEs that have complete coverage across all
four models and a specified number of runs.

Provides:
  - merge_quality_eval()   : public entry point for main.py (merge command)
"""

import re
import pandas as pd


def extract_run_number(run_str):
    """
    Extract an integer run number from a string like 'GAN_02' or 'VAE_1'.
    Returns the number as int, or None if no match.
    """
    match = re.search(r"_(\d+)$", str(run_str))
    return int(match.group(1)) if match else None


def merge_quality_eval(quality_csv, eval_csv, output_file,
                       required_models=None, required_runs=10):
    """
    Merge quality assessment and evaluation CSV files.

    Parameters
    ----------
    quality_csv : str
        Path to the quality assessment CSV (e.g., all_quality_assessments.csv).
        Must contain columns: `real_file`, `run`, `model_type`.
    eval_csv : str
        Path to the evaluation CSV (e.g., evaluation_results.csv).
        Must contain columns: `GSE`, `Model`, `RunID`.
    output_file : str
        Path where the merged (and optionally filtered) CSV will be saved.
    required_models : set of str or None, default None
        If given, only GSEs with at least one run for every model in this set
        are kept. If None, defaults to {"GAN", "WGAN-GP", "diffusion", "VAE"}.
    required_runs : int, default 10
        If given, only GSEs where each required model has exactly this many
        runs are kept. Set to None to disable the per‑model run count filter.

    Returns
    -------
    pd.DataFrame
        The merged (and filtered) DataFrame.
    """
    # ----- 1. Load quality data -----
    df_q = pd.read_csv(quality_csv)
    if 'real_file' not in df_q.columns or 'run' not in df_q.columns:
        raise ValueError("Quality CSV must contain 'real_file' and 'run' columns.")
    # Extract GSE ID from the real_file name (remove .csv/.xlsx extension)
    df_q['GSE'] = df_q['real_file'].str.replace(r"\.(csv|xlsx)$", "", regex=True)
    df_q['run_number'] = df_q['run'].apply(extract_run_number)
    # Drop rows where run_number could not be parsed
    df_q = df_q.dropna(subset=['run_number'])
    df_q['run_number'] = df_q['run_number'].astype(int)

    # ----- 2. Load evaluation data -----
    df_e = pd.read_csv(eval_csv)
    # Rename columns to match quality data's naming convention
    if 'Model' in df_e.columns:
        df_e = df_e.rename(columns={'Model': 'model_type'})
    if 'RunID' in df_e.columns:
        df_e = df_e.rename(columns={'RunID': 'run_number'})
    if 'model_type' not in df_e.columns or 'run_number' not in df_e.columns:
        raise ValueError("Evaluation CSV must contain 'Model' (or 'model_type') and 'RunID' (or 'run_number').")
    df_e['run_number'] = df_e['run_number'].astype(int)

    # ----- 3. Merge on GSE, model_type, run_number -----
    merged = pd.merge(df_q, df_e, on=['GSE', 'model_type', 'run_number'], how='inner')
    if merged.empty:
        raise RuntimeError("No common records found between quality and evaluation data. Check GSE/run IDs.")

    # ----- 4. Optional filtering for complete model/run coverage -----
    if required_models is not None or required_runs is not None:
        if required_models is None:
            required_models = {'GAN', 'WGAN-GP', 'diffusion', 'VAE'}

        valid_gses = []
        for gse, grp in merged.groupby('GSE'):
            model_counts = grp['model_type'].value_counts()
            # Check that all required models are present
            if not set(model_counts.index).issuperset(required_models):
                continue
            # Check run counts if specified
            if required_runs is not None:
                if not (model_counts.loc[list(required_models)] == required_runs).all():
                    continue
            valid_gses.append(gse)
        if not valid_gses:
            raise RuntimeError("No GSE with the required model/run coverage after filtering.")
        merged = merged[merged['GSE'].isin(valid_gses)]

    # ----- 5. Save and return -----
    merged.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file} ({merged.shape[0]} rows)")
    return merged
