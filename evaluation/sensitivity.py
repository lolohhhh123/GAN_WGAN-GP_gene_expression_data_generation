"""
Global weight sensitivity analysis for the composite quality score.

Varies each of the five weights (mean_correlation, 1‑RMSE penalty, variance_correlation,
1‑correlation RMSE penalty, 1‑PCA distance penalty) while keeping the other weights
proportionally renormalised to sum to 1. Plots the resulting raw quality score per model
and saves the raw data.

Provides:
  - run_sensitivity_analysis(results_csv, output_dir) : public entry point for main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Default base weights (must sum to 1)
DEFAULT_WEIGHTS = np.array([0.3, 0.2, 0.2, 0.2, 0.1])

# Weight names (for labels and file names)
WEIGHT_NAMES = [
    'mean_correlation',
    'mean_rmse_penalty',
    'variance_correlation',
    'corr_rmse_penalty',
    'pca_dist_penalty'
]


def _compute_penalties_if_needed(df):
    """
    Ensure the DataFrame contains rmse_penalty, corr_penalty, and pca_penalty.
    If missing, compute them from raw metrics using default thresholds.
    """
    if 'rmse_penalty' not in df.columns:
        df['rmse_penalty'] = np.minimum(df['mean_rmse'] / 10.0, 1.0)
    if 'corr_penalty' not in df.columns:
        df['corr_penalty'] = np.minimum(df['correlation_structure_rmse'] / 0.5, 1.0)
    if 'pca_penalty' not in df.columns:
        df['pca_penalty'] = np.minimum(df['pca_center_distance'] / 10.0, 1.0)
    return df


def _compute_raw_score(df, weights):
    """Compute raw (unclipped) quality score from components and weights."""
    return (weights[0] * df['mean_correlation'] +
            weights[1] * (1 - df['rmse_penalty']) +
            weights[2] * df['variance_correlation'] +
            weights[3] * (1 - df['corr_penalty']) +
            weights[4] * (1 - df['pca_penalty']))


def run_sensitivity_analysis(results_csv, output_dir, weight_range=(0.0, 1.0, 21)):
    """
    Run global weight sensitivity analysis on a merged quality+evaluation CSV.

    Parameters
    ----------
    results_csv : str
        Path to the CSV file (must contain at least 'model_type', plus the five
        component columns: mean_correlation, mean_rmse, variance_correlation,
        correlation_structure_rmse, pca_center_distance). If penalty columns are missing
        they will be computed automatically.
    output_dir : str
        Directory where sensitivity plots and raw CSVs will be saved.
    weight_range : tuple (start, stop, num)
        Values for each weight to sweep. Default is (0.0, 1.0, 21) -> 0.0 to 1.0 in
        steps of 0.05.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    if 'model_type' not in df.columns:
        raise ValueError("Input CSV must contain a 'model_type' column.")

    print(f"Models found: {df['model_type'].unique()}")

    # Ensure penalty columns exist
    df = _compute_penalties_if_needed(df)

    # Five component columns for the weighted sum
    comp1 = df['mean_correlation'].values
    comp2 = 1 - df['rmse_penalty'].values
    comp3 = df['variance_correlation'].values
    comp4 = 1 - df['corr_penalty'].values
    comp5 = 1 - df['pca_penalty'].values
    comp_vals = np.column_stack([comp1, comp2, comp3, comp4, comp5])

    weight_values = np.linspace(*weight_range)

    for idx in range(5):
        print(f"Processing sensitivity for {WEIGHT_NAMES[idx]}...")
        scores = []
        for w in weight_values:
            # Build weight vector with current weight w for component idx,
            # other weights scaled so total sum = 1.
            w_vec = DEFAULT_WEIGHTS.copy()
            other_sum = 1.0 - w
            if other_sum > 0:
                # Scale all weights first to have sum = other_sum except idx
                scale = other_sum / (1.0 - w_vec[idx]) if (1.0 - w_vec[idx]) > 0 else 0
                w_vec = w_vec * scale
                w_vec[idx] = w
            else:
                w_vec = np.zeros(5)
                w_vec[idx] = w
            # Force sum to 1 (float precision)
            w_vec = w_vec / w_vec.sum()
            scores.append(comp_vals @ w_vec)
        scores = np.array(scores).T  # (n_samples, n_weights)

        # Save raw data
        out_df = pd.DataFrame(scores, columns=[f'w_{v:.3f}' for v in weight_values])
        out_df['model_type'] = df['model_type'].values
        out_df.to_csv(os.path.join(output_dir, f'sensitivity_raw_{WEIGHT_NAMES[idx]}.csv'), index=False)

        # Prepare long‑form DataFrame for plotting
        plot_records = []
        for i, w in enumerate(weight_values):
            for j in range(scores.shape[0]):
                plot_records.append({
                    'weight': w,
                    'score': scores[j, i],
                    'model_type': df['model_type'].iloc[j]
                })
        plot_df = pd.DataFrame(plot_records)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=plot_df, x='weight', y='score', hue='model_type',
                     estimator='mean', errorbar=('ci', 95), err_style='band')
        plt.xlabel(f'Weight for {WEIGHT_NAMES[idx]}')
        plt.ylabel('Overall quality score (raw)')
        plt.title(f'Sensitivity to {WEIGHT_NAMES[idx]}')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sensitivity_{WEIGHT_NAMES[idx]}.png'), dpi=150)
        plt.close()
        print(f"  Saved plot and data for {WEIGHT_NAMES[idx]}")

    print("Sensitivity analysis complete.")
