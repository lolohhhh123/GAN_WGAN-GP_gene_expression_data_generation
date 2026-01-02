# Alzheimer's Disease gene expression data Synthetic Data Generator

## Overview
This repository contains a Synthetic Data Generator for generating synthetic gene expression data for Alzheimer's Disease research. It implements both WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty) and traditional GAN approaches to create statistically similar synthetic data while preserving the biological characteristics of the original datasets.

## Features
- **Dual GAN Support**: Both WGAN-GP (TensorFlow) and traditional GAN (PyTorch) implementations
- **Automatic Preprocessing**: Robust data normalization and dimensionality reduction
- **Statistical Evaluation**: Kolmogorov-Smirnov tests for distribution comparison
- **Multiple Runs**: Supports repeated generation for robust statistical analysis
- **Flexible Output**: Generates separate Control and AD samples

## Installation
```bash
git clone https://github.com/yourusername/ad-synthetic-data-generator.git
cd ad-synthetic-data-generator
pip install -r requirements.txt
