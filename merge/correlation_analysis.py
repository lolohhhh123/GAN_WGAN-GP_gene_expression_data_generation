"""
Correlation analysis between evaluation metrics and quality score.

Computes Pearson and Spearman correlations of all numeric columns
against a target score, optionally broken down by model type.

Provides:
  - run_correlation_analysis()   : public entry point for main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def run_correlation_analysis(input_csv, output_dir,
                             target='overall_quality_score',
                             group_col='model_type'):
    """
    Compute correlations of numeric columns in a merged quality+evaluation CSV
    against a target quality score, and produce summary tables and plots.

    Parameters
    ----------
    input_csv : str
        Path to the merged CSV (e.g., from merge_quality_eval).
        Must contain the target column and, if grouped analysis is desired,
        a column with model names (default 'model_type').
    output_dir : str
        Directory where correlation tables and heatmaps will be saved.
    target : str, default 'overall_quality_score'
        Column name of the target variable.
    group_col : str or None, default 'model_type'
        Column name for grouping models. If None, per‑model analysis is skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in numeric_cols:
        raise ValueError(f"Target column '{target}' not found in numeric columns.")

    # ----- 1. Overall correlations -----
    pearson = {}
    spearman = {}
    for col in numeric_cols:
        if col == target:
            continue
        valid = df[[col, target]].dropna()
        if len(valid) > 1:
            pearson[col] = pearsonr(valid[col], valid[target])[0]
            spearman[col] = spearmanr(valid[col], valid[target])[0]

    summary = pd.DataFrame({
        'feature': list(pearson.keys()),
        'pearson_r': list(pearson.values()),
        'spearman_r': list(spearman.values())
    })
    summary = summary.sort_values('pearson_r', ascending=False, key=abs)
    summary_path = os.path.join(output_dir, 'feature_correlations.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Saved correlation summary to {summary_path}")

    # ----- 2. Full correlation heatmap -----
    corr_matrix = df[numeric_cols].corr(method='pearson')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Pearson Correlation Matrix (all numeric features)')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    # ----- 3. Per‑model breakdown (if group_col is present) -----
    if group_col and group_col in df.columns:
        models = df[group_col].unique()
        grouped = []
        for model in models:
            subset = df[df[group_col] == model]
            numeric = subset.select_dtypes(include=[np.number])
            if target in numeric.columns and len(numeric) > 1:
                corr_with_target = numeric.corr()[target].drop(target, errors='ignore')
                for feat, val in corr_with_target.items():
                    grouped.append({
                        group_col: model,
                        'feature': feat,
                        'pearson_r': val
                    })
        if grouped:
            grouped_df = pd.DataFrame(grouped)
            grouped_path = os.path.join(output_dir, 'correlations_by_model.csv')
            grouped_df.to_csv(grouped_path, index=False)
            print(f"Saved per‑model correlations to {grouped_path}")
    else:
        print(f"Group column '{group_col}' not found; skipping per‑model analysis.")

    print("Correlation analysis completed.")
