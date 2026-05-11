"""
Reusable plotting utilities for the synthetic-genome-eval project.
All functions use the 'Agg' backend and are safe for headless servers.

Functions:
    save_boxplot()      – create a boxplot and save to file.
    save_heatmap()      – save a heatmap of mean/median values.
    write_summary_txt() – append metric averages/medians as a text report.
    generate_model_metric_plots() – convenience wrapper for boxplots + heatmap
                                    across multiple metrics for different models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')               # non‑interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def save_boxplot(data, x, y, output_path, title=None, xlabel=None, ylabel=None,
                 figsize=(10, 6), rotation=45, dpi=150):
    """
    Create a boxplot and save it to disk.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Column name for the x‑axis (e.g., 'Model').
    y : str
        Column name for the y‑axis (e.g., 'MMD_overall').
    output_path : str
        Full path to the output PNG file.
    title, xlabel, ylabel : str, optional
    figsize : tuple, default (10, 6)
    rotation : int, default 45
        Rotation angle for x‑tick labels.
    dpi : int, default 150
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x, y=y)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_heatmap(data, output_path, annot=True, fmt='.3f', cmap='viridis',
                 figsize=None, title=None, dpi=150):
    """
    Save a heatmap of aggregated values (e.g., mean metrics per model).

    Parameters
    ----------
    data : pd.DataFrame
        Index = model names, columns = metric names.
    output_path : str
    annot : bool, default True
        Whether to write the numeric values in each cell.
    fmt : str, default '.3f'
        Format string for annotations.
    cmap : str, default 'viridis'
    figsize : tuple, optional
        Automatically adjusted if None based on number of rows.
    title : str, optional
    dpi : int, default 150
    """
    if figsize is None:
        figsize = (8, max(4, len(data) * 0.5 + 2))
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def write_summary_txt(output_path, results_df, metric_cols, group_col='Model'):
    """
    Append a text summary of average and median metrics per group.

    Parameters
    ----------
    output_path : str
        Text file to write (will be opened in append mode).
    results_df : pd.DataFrame
    metric_cols : list of str
    group_col : str, default 'Model'
    """
    with open(output_path, 'a') as f:
        f.write("\nAverage metrics per model:\n")
        f.write(results_df.groupby(group_col)[metric_cols].mean().to_string())
        f.write("\n\nMedian metrics per model:\n")
        f.write(results_df.groupby(group_col)[metric_cols].median().to_string())
        f.write("\n" + "=" * 60 + "\n")


def generate_model_metric_plots(results_df, output_dir, metric_cols,
                                group_col='Model', prefix=''):
    """
    Convenience wrapper that produces:
      - one boxplot per metric (saved as '{prefix}boxplot_{metric}.png')
      - a heatmap of mean metric values per model
      - a summary of averages/medians in 'summary.txt'.

    Parameters
    ----------
    results_df : pd.DataFrame
    output_dir : str
    metric_cols : list of str
        Numeric columns to plot (e.g., ['MMD_overall', 'FID_overall']).
    group_col : str, default 'Model'
    prefix : str, default ''
        Optional prefix for the output filenames.
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric in metric_cols:
        if metric in results_df.columns and results_df[metric].notna().any():
            save_boxplot(
                results_df, x=group_col, y=metric,
                output_path=os.path.join(output_dir, f'{prefix}boxplot_{metric}.png'),
                title=f'{metric} per {group_col}',
                xlabel=group_col, ylabel=metric
            )

    avail = [m for m in metric_cols if m in results_df.columns]
    if avail:
        avg = results_df.groupby(group_col)[avail].mean()
        save_heatmap(
            avg,
            output_path=os.path.join(output_dir, f'{prefix}heatmap_avg_metrics.png'),
            title=f'Average Metrics per {group_col}'
        )

    write_summary_txt(
        os.path.join(output_dir, 'summary.txt'),
        results_df, avail, group_col
    )
