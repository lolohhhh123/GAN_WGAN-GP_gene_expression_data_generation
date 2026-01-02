# Alzheimer's Disease gene expression data Synthetic Data Generator

## Overview
This repository contains a Synthetic Data Generator for generating synthetic gene expression data for Alzheimer's Disease research. It implements both WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty) and traditional GAN approaches with quality assessment with wPCA scoring to create high quality and statistically similar synthetic data while preserving the biological characteristics of the original datasets.

## Features
- **Dual GAN Support**: Both WGAN-GP (TensorFlow) and traditional GAN (PyTorch) implementations
- **Automatic Preprocessing**: Robust data normalization and dimensionality reduction
- **Statistical Evaluation**: Kolmogorov-Smirnov tests for distribution comparison
- **Multiple Runs**: Supports repeated generation for robust statistical analysis
- **Flexible Output**: Generates separate Control and AD samples
- **wPCA Score**: Weighted PCA score that considers distribution differences across principal components
- **Multiple Metrics**: Includes KS statistics, distribution metrics, correlation analysis
- **Visualization**: PCA scatter plots, distribution comparisons, quality score summaries
- **Batch Processing**: Assess multiple datasets simultaneously
- 
## Installation
```bash
git clone https://github.com/yourusername/ad-synthetic-data-generator.git
cd ad-synthetic-data-generator
pip install -r requirements.txt

## Single Dataset Assessment
```python
from data_quality_assessment import DataQualityAssessor

assessor = DataQualityAssessor(random_seed=42)
quality_report = assessor.comprehensive_data_quality_assessment(
    real_data_path, synthetic_data_path, output_dir
)

