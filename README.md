# Alzheimer's Disease gene expression data Synthetic Data Generator

## Overview
This repository contains a Synthetic Data Generator for generating gene expression data for Alzheimer's Disease research. It implements GAN，WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty)，diffusion and VAE models approaches with quality assessment with Weighted quality scores (WQS) to create high quality and statistically similar synthetic data while preserving the biological characteristics of the original datasets.

## Features
- **Automatic Preprocessing**: Robust data normalization and dimensionality reduction
- **Multiple Runs**: Supports repeated generation for robust statistical analysis
- **Flexible Output**: Generates Control and AD samples
- **WQS**: WQS considers distribution differences across 5 different components
- **Multiple Metrics**: Includes KS statistics, distribution metrics, correlation analysis
- **Visualization**: PCA scatter plots, distribution comparisons, quality score summaries
- **Batch Processing**: Assess multiple datasets simultaneously

## Installation
```bash
git clone https://github.com/lolohhhh123/Gene_expression_data_generation
cd Gene_expression_data_generation
pip install -r requirements.txt

How to generate data:
python main.py generate --model VAE --input_dir data/ --output_dir results/generation --runs 10

How to evaluate data quality:
python main.py evaluate --method all --real_dir data/ --synth_root results/generation --output_dir results/eval

How to merge data:
python main.py merge --quality_csv ... --eval_csv ... --output_file merged.csv

How to run sensitivity check:
python main.py sensitivity --results_csv merged.csv --output_dir sensitivity/
