#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate synthetic data from multiple generative models against original data.
Computes MMD and Fréchet distance (FID) overall and per group (Control/AD).
CPU-only, multiprocessing-safe. Adds RunID column to track specific subfolder.
"""

import re, numpy as np, pandas as pd, warnings
"""Load CSV/Excel and return {'data':array(rows x cols), 'sample_names':list, 'gene_names':list}."""
def load_data_file(filepath, transpose=True, na_threshold=0.5, fill_na=True):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, header=0)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, header=0)
    else:
        raise ValueError(f"Unsupported format: {filepath}")

    gene_names = df.iloc[:, 0].astype(str).values
    sample_names = df.columns[1:].values
    data_mat = df.iloc[:, 1:].values.astype(np.float64)

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

    if len(np.unique(colnames)) < len(colnames):
        df_agg = pd.DataFrame(data_mat, index=rownames, columns=colnames)
        df_agg = df_agg.T.groupby(level=0).mean().T
        data_mat = df_agg.values
        colnames = df_agg.columns.values

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

    return {
        'data': data_mat,
        'sample_names': rownames,
        'gene_names': colnames
    }

def infer_labels_from_names(sample_names):
    """Return np.array of 0/1 based on 'Control'/'AD' prefix."""
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
    """For synthetic data: assume all Control or all AD come first, return labels array and split index."""
    seen_base = set()
    for i, name in enumerate(sample_names):
        print("name",name)
        name = str(name)
        if "_synth_" in name:
            if name.startswith("AD"):
                base = name[2:]  # remove first seven characters 'Control' / last two characters '.1'
                base = "Control" + base
            else:
               #,.base = name[7:]
                base = name
                #base = "AD" + base
            if base in seen_base:
                # Found first duplicate (with _synth_ / .1 suffix)
                labels = np.array([0] * i + [1] * (len(sample_names) - i))
                return labels, i
            else:
                # This case shouldn't happen if order is correct, but handle anyway
                seen_base.add(base)
        elif "Generated_Sample" in name:
            if name.endswith(".1"):
                base = name[:-2]  # remove first seven characters 'Control' / last two characters '.1'
            else:
                base = name
            if base in seen_base:
                labels = np.array([0] * i + [1] * (len(sample_names) - i))
                return labels, i
            else:
                seen_base.add(base)
                print("base:",base)
    return None, None

def align_on_common_genes(orig_data, orig_genes, gen_data, gen_genes):
    """Return aligned arrays for common genes."""
    common = list(set(orig_genes) & set(gen_genes))
    if not common:
        raise ValueError("No common genes found")
    o_idx = [i for i, g in enumerate(orig_genes) if g in common]
    g_idx = [i for i, g in enumerate(gen_genes) if g in common]
    return orig_data[:, o_idx], gen_data[:, g_idx], common
