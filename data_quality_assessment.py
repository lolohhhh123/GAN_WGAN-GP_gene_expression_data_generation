"""
Comprehensive Data Quality Assessment Module
Includes wPCA scoring, distribution metrics, and visualization
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import umap

# Suppress warnings
warnings.filterwarnings('ignore')

class DataQualityAssessor:
    """Comprehensive data quality assessment with wPCA scoring"""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data quality assessor"""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    @staticmethod
    def safe_convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Safely convert data to numeric type"""
        try:
            # Method 1: Try direct conversion
            df_processed = df.apply(pd.to_numeric, errors='coerce')
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
            df_processed = df_processed.fillna(0)
            return df_processed
        except Exception:
            # Method 2: Column-wise conversion
            df_processed = df.copy()
            for col in df.columns:
                try:
                    col_data = df[col]
                    if isinstance(col_data, pd.Series):
                        df_processed[col] = pd.to_numeric(col_data, errors='coerce')
                    else:
                        col_array = np.array(col_data).flatten()
                        df_processed[col] = pd.to_numeric(col_array, errors='coerce')
                except Exception:
                    df_processed[col] = 0
            
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
            df_processed = df_processed.fillna(0)
            return df_processed
    
    @staticmethod
    def safe_extract_feature_values(data: pd.DataFrame, feature: str) -> np.ndarray:
        """Safely extract feature values as 1D array"""
        try:
            if isinstance(data, pd.DataFrame):
                feature_data = data[feature]
            else:
                feature_data = data
            
            values = np.array(feature_data).flatten()
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            return values
        except Exception:
            return np.array([0.0])
    
    def calculate_dataset_ks(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                            max_features: int = 1000) -> Dict[str, float]:
        """Calculate KS statistics for entire dataset"""
        try:
            # Safely convert data to numeric
            real_data = self.safe_convert_to_numeric(real_data)
            synthetic_data = self.safe_convert_to_numeric(synthetic_data)
            
            # Limit features if too many
            if real_data.shape[1] > max_features:
                selected_features = np.random.choice(
                    real_data.columns, size=max_features, replace=False
                )
                real_data = real_data[selected_features]
                synthetic_data = synthetic_data[selected_features]
            
            ks_stats = []
            ks_pvalues = []
            failed_features = []
            
            # KS test for each feature
            for feature in real_data.columns:
                try:
                    real_values = self.safe_extract_feature_values(real_data, feature)
                    synthetic_values = self.safe_extract_feature_values(synthetic_data, feature)
                    
                    # Check data validity
                    if (len(real_values) == 0 or len(synthetic_values) == 0 or
                        not np.isfinite(real_values).all() or not np.isfinite(synthetic_values).all()):
                        failed_features.append(feature)
                        continue
                    
                    # Handle constant data
                    if (np.all(real_values == real_values[0]) and 
                        np.all(synthetic_values == synthetic_values[0])):
                        if real_values[0] == synthetic_values[0]:
                            ks_stat, p_value = 0.0, 1.0
                        else:
                            ks_stat, p_value = 1.0, 0.0
                    else:
                        ks_stat, p_value = ks_2samp(real_values, synthetic_values)
                    
                    ks_stats.append(ks_stat)
                    ks_pvalues.append(p_value)
                        
                except Exception:
                    failed_features.append(feature)
                    continue
            
            if len(ks_stats) == 0:
                return {
                    'ks_mean': np.nan,
                    'ks_median': np.nan,
                    'ks_max': np.nan,
                    'ks_min': np.nan,
                    'ks_std': np.nan,
                    'ks_significant_features': 0,
                    'ks_pvalue_mean': np.nan,
                    'total_features_tested': 0,
                    'failed_features': len(failed_features)
                }
            
            # Calculate summary statistics
            ks_stats = np.array(ks_stats)
            ks_pvalues = np.array(ks_pvalues)
            significant_features = np.sum(ks_pvalues < 0.05)
            
            ks_results = {
                'ks_mean': np.mean(ks_stats),
                'ks_median': np.median(ks_stats),
                'ks_max': np.max(ks_stats),
                'ks_min': np.min(ks_stats),
                'ks_std': np.std(ks_stats),
                'ks_significant_features': significant_features,
                'ks_pvalue_mean': np.mean(ks_pvalues),
                'total_features_tested': len(ks_stats),
                'failed_features': len(failed_features)
            }
            
            return ks_results
        
        except Exception as e:
            print(f"Error calculating KS statistics: {e}")
            return {
                'ks_mean': np.nan,
                'ks_median': np.nan,
                'ks_max': np.nan,
                'ks_min': np.nan,
                'ks_std': np.nan,
                'ks_significant_features': 0,
                'ks_pvalue_mean': np.nan,
                'total_features_tested': 0,
                'failed_features': 'unknown'
            }
    
    def load_and_prepare_data(self, real_data_path: Path, synthetic_data_path: Path) -> Tuple:
        """Load and prepare real and synthetic data"""
        try:
            # Read data
            if real_data_path.suffix == '.csv':
                real_df = pd.read_csv(real_data_path)
            else:
                real_df = pd.read_excel(real_data_path)
            
            if synthetic_data_path.suffix == '.csv':
                synthetic_df = pd.read_csv(synthetic_data_path)
            else:
                synthetic_df = pd.read_excel(synthetic_data_path)
            
            # Determine data orientation
            if real_df.shape[0] > real_df.shape[1] * 2:
                # Genes in rows, samples in columns - transpose
                if real_df.columns[0] != 'Unnamed: 0':
                    real_df.set_index(real_df.columns[0], inplace=True)
                    synthetic_df.set_index(synthetic_df.columns[0], inplace=True)
                
                real_df_t = real_df.T
                synthetic_df_t = synthetic_df.T
            else:
                # Genes in columns, samples in rows
                if real_df.columns[0] != 'Unnamed: 0':
                    real_df.set_index(real_df.columns[0], inplace=True)
                    synthetic_df.set_index(synthetic_df.columns[0], inplace=True)
                
                real_df_t = real_df
                synthetic_df_t = synthetic_df
            
            # Reset indices
            real_df_t.reset_index(drop=True, inplace=True)
            synthetic_df_t.reset_index(drop=True, inplace=True)
            
            # Ensure string column names
            real_df_t.columns = real_df_t.columns.astype(str)
            synthetic_df_t.columns = synthetic_df_t.columns.astype(str)
            
            # Find common genes
            common_genes = real_df_t.columns.intersection(synthetic_df_t.columns)
            
            if len(common_genes) == 0:
                # Try more flexible matching
                real_genes_lower = set([g.lower().strip() for g in real_df_t.columns])
                synthetic_genes_lower = set([g.lower().strip() for g in synthetic_df_t.columns])
                common_genes_lower = real_genes_lower.intersection(synthetic_genes_lower)
                
                if len(common_genes_lower) > 0:
                    real_gene_map = {g.lower().strip(): g for g in real_df_t.columns}
                    synthetic_gene_map = {g.lower().strip(): g for g in synthetic_df_t.columns}
                    common_genes = [real_gene_map[g] for g in common_genes_lower]
                else:
                    common_genes = real_df_t.columns.union(synthetic_df_t.columns)
            
            # Select common genes
            real_df_t = real_df_t[common_genes]
            synthetic_df_t = synthetic_df_t[common_genes]
            
            # Convert to numeric
            real_df_t = self.safe_convert_to_numeric(real_df_t)
            synthetic_df_t = self.safe_convert_to_numeric(synthetic_df_t)
            
            return real_df_t, synthetic_df_t, common_genes
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def pca_comparison(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                      n_components: int = 50) -> Tuple:
        """PCA comparison between real and synthetic data"""
        try:
            # Add labels
            real_data_copy = real_data.copy()
            synthetic_data_copy = synthetic_data.copy()
            
            real_data_copy['data_type'] = 'real'
            synthetic_data_copy['data_type'] = 'synthetic'
            combined_data = pd.concat([real_data_copy, synthetic_data_copy], axis=0, ignore_index=True)
            
            # Separate features and labels
            X = combined_data.drop('data_type', axis=1)
            y = combined_data['data_type']
            
            # Ensure numeric data
            X = X.apply(pd.to_numeric, errors='coerce')
            
            # Check data validity
            if X.isna().all().all() or X.isnull().all().all():
                return None, None, None, None
            
            # Handle NaN values
            X = X.fillna(0)
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine actual component count
            n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
            if n_components <= 0:
                return None, None, None, None
            
            # PCA dimensionality reduction
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create results DataFrame
            pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
            pca_df['data_type'] = y.values
            pca_df.reset_index(drop=True, inplace=True)
            
            # Calculate explained variance ratio
            explained_variance = pca.explained_variance_ratio_
            
            return pca_df, explained_variance, pca, scaler
        
        except Exception as e:
            print(f"Error in PCA comparison: {e}")
            return None, None, None, None
    
    def calculate_distribution_metrics(self, real_data: pd.DataFrame, 
                                     synthetic_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate multiple distribution similarity metrics"""
        metrics = {}
        
        try:
            # Ensure numeric data
            real_data = real_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            synthetic_data = synthetic_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # 1. Mean differences
            mean_real = real_data.mean(axis=0)
            mean_synthetic = synthetic_data.mean(axis=0)
            
            if not np.isnan(mean_real).all() and not np.isnan(mean_synthetic).all():
                valid_idx = ~(np.isnan(mean_real) | np.isnan(mean_synthetic))
                if valid_idx.any():
                    metrics['mean_correlation'] = np.corrcoef(
                        mean_real[valid_idx], mean_synthetic[valid_idx]
                    )[0, 1]
                    metrics['mean_rmse'] = np.sqrt(np.mean(
                        (mean_real[valid_idx] - mean_synthetic[valid_idx]) ** 2
                    ))
                else:
                    metrics['mean_correlation'] = np.nan
                    metrics['mean_rmse'] = np.nan
            else:
                metrics['mean_correlation'] = np.nan
                metrics['mean_rmse'] = np.nan
            
            # 2. Variance differences
            var_real = real_data.var(axis=0)
            var_synthetic = synthetic_data.var(axis=0)
            
            if not np.isnan(var_real).all() and not np.isnan(var_synthetic).all():
                valid_idx = ~(np.isnan(var_real) | np.isnan(var_synthetic))
                if valid_idx.any():
                    metrics['variance_correlation'] = np.corrcoef(
                        var_real[valid_idx], var_synthetic[valid_idx]
                    )[0, 1]
                else:
                    metrics['variance_correlation'] = np.nan
            else:
                metrics['variance_correlation'] = np.nan
            
            # 3. Correlation structure differences
            try:
                n_genes = min(1000, real_data.shape[1])
                if n_genes > 10:
                    selected_genes = real_data.columns[:n_genes]
                    corr_real = np.corrcoef(real_data[selected_genes].T)
                    corr_synthetic = np.corrcoef(synthetic_data[selected_genes].T)
                    metrics['correlation_rmse'] = np.sqrt(np.mean((corr_real - corr_synthetic) ** 2))
                else:
                    metrics['correlation_rmse'] = np.nan
            except:
                metrics['correlation_rmse'] = np.nan
            
            # 4. Principal component distances
            n_components = min(10, real_data.shape[0], real_data.shape[1], 
                              synthetic_data.shape[0], synthetic_data.shape[1])
            if n_components > 1:
                try:
                    pca = PCA(n_components=n_components)
                    pca_real = pca.fit_transform(real_data)
                    pca_synthetic = pca.transform(synthetic_data)
                    
                    # Calculate distance between PCA centers
                    center_real = pca_real.mean(axis=0)
                    center_synthetic = pca_synthetic.mean(axis=0)
                    metrics['pca_center_distance'] = np.linalg.norm(center_real - center_synthetic)
                except:
                    metrics['pca_center_distance'] = np.nan
            else:
                metrics['pca_center_distance'] = np.nan
            
            return metrics
        
        except Exception as e:
            print(f"Error calculating distribution metrics: {e}")
            return {
                'mean_correlation': np.nan, 'mean_rmse': np.nan, 
                'variance_correlation': np.nan, 'correlation_rmse': np.nan,
                'pca_center_distance': np.nan
            }
    
    def visualize_pca_comparison(self, pca_df: pd.DataFrame, explained_variance: np.ndarray, 
                               output_path: Path) -> List[float]:
        """Visualize PCA comparison results"""
        try:
            # Set seaborn style
            sns.set_theme(style="whitegrid")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. PC1 vs PC2 scatter plot
            if 'PC1' in pca_df.columns and 'PC2' in pca_df.columns:
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='data_type', alpha=0.6, ax=axes[0, 0])
                axes[0, 0].set_title('PCA: PC1 vs PC2')
                axes[0, 0].legend()
            else:
                axes[0, 0].text(0.5, 0.5, 'No PC1/PC2 data', ha='center', va='center')
                axes[0, 0].set_title('PCA: PC1 vs PC2')
            
            # 2. PC1 distribution comparison
            if 'PC1' in pca_df.columns:
                pc1_data = pca_df[['PC1', 'data_type']].dropna()
                if not pc1_data.empty:
                    sns.kdeplot(data=pc1_data, x='PC1', hue='data_type', ax=axes[0, 1])
                    axes[0, 1].set_title('PC1 Distribution Comparison')
                else:
                    axes[0, 1].text(0.5, 0.5, 'No valid PC1 data', ha='center', va='center')
                    axes[0, 1].set_title('PC1 Distribution Comparison')
            else:
                axes[0, 1].text(0.5, 0.5, 'No PC1 data', ha='center', va='center')
                axes[0, 1].set_title('PC1 Distribution Comparison')
            
            # 3. Explained variance ratio
            if len(explained_variance) > 0:
                cumulative_variance = np.cumsum(explained_variance)
                axes[1, 0].plot(range(1, len(explained_variance) + 1), cumulative_variance, 'b-')
                axes[1, 0].set_xlabel('Number of Principal Components')
                axes[1, 0].set_ylabel('Cumulative Explained Variance Ratio')
                axes[1, 0].set_title('Cumulative Explained Variance')
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, 'No variance data', ha='center', va='center')
                axes[1, 0].set_title('Cumulative Explained Variance')
            
            # 4. Distribution differences by principal component
            n_components_show = min(5, pca_df.shape[1] - 1)
            component_differences = []
            
            if n_components_show > 0:
                for i in range(n_components_show):
                    pc_col = f'PC{i+1}'
                    if pc_col in pca_df.columns:
                        real_pc = pca_df[pca_df['data_type'] == 'real'][pc_col].dropna()
                        synthetic_pc = pca_df[pca_df['data_type'] == 'synthetic'][pc_col].dropna()
                        
                        if len(real_pc) > 0 and len(synthetic_pc) > 0:
                            ks_stat, _ = stats.ks_2samp(real_pc, synthetic_pc)
                            component_differences.append(ks_stat)
                        else:
                            component_differences.append(np.nan)
                    else:
                        component_differences.append(np.nan)
                
                # Filter NaN values
                valid_differences = [x for x in component_differences if not np.isnan(x)]
                if valid_differences:
                    axes[1, 1].bar(range(1, len(valid_differences) + 1), valid_differences)
                    axes[1, 1].set_xlabel('Principal Component')
                    axes[1, 1].set_ylabel('KS Statistic')
                    axes[1, 1].set_title('Distribution Difference by Principal Component')
                    axes[1, 1].grid(True)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No valid component data', ha='center', va='center')
                    axes[1, 1].set_title('Distribution Difference by Principal Component')
            else:
                axes[1, 1].text(0.5, 0.5, 'No components to show', ha='center', va='center')
                axes[1, 1].set_title('Distribution Difference by Principal Component')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return component_differences
        
        except Exception as e:
            print(f"Error visualizing PCA comparison: {e}")
            return []
    
    def calculate_wpca_score(self, pca_df: pd.DataFrame, explained_variance: np.ndarray) -> float:
        """Calculate weighted PCA (wPCA) score"""
        try:
            if len(explained_variance) == 0:
                return np.nan
            
            # Calculate KS statistics for each principal component
            component_ks = []
            for i in range(len(explained_variance)):
                pc_col = f'PC{i+1}'
                if pc_col in pca_df.columns:
                    real_pc = pca_df[pca_df['data_type'] == 'real'][pc_col].dropna()
                    synthetic_pc = pca_df[pca_df['data_type'] == 'synthetic'][pc_col].dropna()
                    
                    if len(real_pc) > 0 and len(synthetic_pc) > 0:
                        ks_stat, _ = stats.ks_2samp(real_pc, synthetic_pc)
                        component_ks.append(ks_stat)
                    else:
                        component_ks.append(1.0)  # Maximum dissimilarity
                else:
                    component_ks.append(1.0)
            
            # Calculate weighted average using explained variance as weights
            component_ks = np.array(component_ks)
            weights = explained_variance[:len(component_ks)]
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Calculate wPCA score (1 - weighted KS, so higher is better)
            wpca_score = 1 - np.sum(weights * component_ks)
            
            return max(0, min(1, wpca_score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            print(f"Error calculating wPCA score: {e}")
            return np.nan
    
    def comprehensive_data_quality_assessment(self, real_data_path: Path, 
                                            synthetic_data_path: Path, 
                                            output_dir: Path) -> Tuple:
        """Comprehensive data quality assessment"""
        # Load data
        real_data, synthetic_data, common_genes = self.load_and_prepare_data(
            real_data_path, synthetic_data_path
        )
        
        if real_data is None or synthetic_data is None:
            return None, None
        
        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # PCA comparison
        pca_results = self.pca_comparison(real_data, synthetic_data)
        if pca_results[0] is None:
            return None, None
            
        pca_df, explained_variance, pca_model, scaler = pca_results
        
        # Calculate distribution metrics
        distribution_metrics = self.calculate_distribution_metrics(real_data, synthetic_data)
        
        # Calculate dataset-wide KS statistics
        ks_metrics = self.calculate_dataset_ks(real_data, synthetic_data)
        
        # Calculate wPCA score
        wpca_score = self.calculate_wpca_score(pca_df, explained_variance)
        
        # Visualize results
        viz_path = output_dir / 'pca_comparison.png'
        component_differences = self.visualize_pca_comparison(pca_df, explained_variance, viz_path)
        
        # Generate quality report
        quality_report = {
            'n_real_samples': real_data.shape[0],
            'n_synthetic_samples': synthetic_data.shape[0],
            'n_common_genes': len(common_genes),
            'pca_explained_variance_ratio': explained_variance.sum() if len(explained_variance) > 0 else 0,
            'mean_correlation': distribution_metrics.get('mean_correlation', np.nan),
            'mean_rmse': distribution_metrics.get('mean_rmse', np.nan),
            'variance_correlation': distribution_metrics.get('variance_correlation', np.nan),
            'correlation_structure_rmse': distribution_metrics.get('correlation_rmse', np.nan),
            'pca_center_distance': distribution_metrics.get('pca_center_distance', np.nan),
            'avg_pc_ks_statistic': np.nanmean(component_differences) if component_differences else np.nan,
            'max_pc_ks_statistic': np.nanmax(component_differences) if component_differences else np.nan,
            'wpca_score': wpca_score
        }
        
        # Add KS metrics to quality report
        quality_report.update(ks_metrics)
        
        # Calculate overall quality score
        quality_score = 0
        weights = 0
        
        if not np.isnan(quality_report['mean_correlation']):
            quality_score += quality_report['mean_correlation'] * 0.2
            weights += 0.2
            
        if not np.isnan(quality_report['variance_correlation']):
            quality_score += quality_report['variance_correlation'] * 0.15
            weights += 0.15
            
        if not np.isnan(wpca_score):
            quality_score += wpca_score * 0.3
            weights += 0.3
            
        if not np.isnan(quality_report['ks_mean']):
            ks_contribution = 1 - min(quality_report['ks_mean'], 1)
            quality_score += ks_contribution * 0.2
            weights += 0.2
            
        if not np.isnan(quality_report['correlation_structure_rmse']):
            normalized_corr_rmse = 1 - min(quality_report['correlation_structure_rmse'] / 0.5, 1)
            quality_score += normalized_corr_rmse * 0.15
            weights += 0.15
        
        if weights > 0:
            quality_report['overall_quality_score'] = max(0, min(1, quality_score / weights))
        else:
            quality_report['overall_quality_score'] = 0
        
        # Save detailed results
        report_df = pd.DataFrame([quality_report])
        report_path = output_dir / 'data_quality_assessment.csv'
        report_df.to_csv(report_path, index=False)
        
        # Save PCA results
        pca_path = output_dir / 'pca_results.csv'
        pca_df.to_csv(pca_path, index=False)
        
        print(f"Quality assessment completed. Overall score: {quality_report['overall_quality_score']:.3f}")
        print(f"wPCA score: {wpca_score:.3f}")
        
        return quality_report, pca_df
    
    def batch_quality_assessment(self, real_data_dir: Path, synthetic_data_dir: Path, 
                               output_base_dir: Path) -> Optional[pd.DataFrame]:
        """Batch quality assessment for multiple datasets"""
        # Create output directory
        output_base_dir.mkdir(exist_ok=True, parents=True)
        
        # Find matching file pairs
        real_files = list(real_data_dir.glob("*.csv"))
        synthetic_files = list(synthetic_data_dir.glob("*.xlsx"))
        
        print(f"Found {len(real_files)} real data files")
        print(f"Found {len(synthetic_files)} synthetic data files")
        
        results = []
        
        for real_file in real_files:
            # Try multiple naming patterns for synthetic files
            name = real_file.stem
            synthetic_patterns = [
                synthetic_data_dir / f"{name}_synthetic_all.xlsx",
                synthetic_data_dir / f"{name}_synthetic_all.csv",
                synthetic_data_dir / f"{name}_generated.xlsx",
            ]
            
            synthetic_file = None
            for pattern in synthetic_patterns:
                if pattern.exists():
                    synthetic_file = pattern
                    break
            
            if synthetic_file is None:
                print(f"No matching synthetic file found for: {name}")
                continue
            
            # Create dataset-specific output directory
            dataset_output_dir = output_base_dir / name
            dataset_output_dir.mkdir(exist_ok=True)
            
            print(f"\nProcessing dataset: {name}")
            print(f"Real data: {real_file}")
            print(f"Synthetic data: {synthetic_file}")
            
            try:
                # Comprehensive quality assessment
                quality_report, _ = self.comprehensive_data_quality_assessment(
                    real_file, synthetic_file, dataset_output_dir
                )
                
                if quality_report is not None:
                    quality_report['dataset'] = name
                    results.append(quality_report)
                
            except Exception as e:
                print(f"Error processing dataset {name}: {e}")
                continue
        
        # Generate summary report
        if results:
            summary_df = pd.DataFrame(results)
            summary_path = output_base_dir / 'quality_assessment_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            
            # Visualize quality scores
            if not summary_df.empty and 'overall_quality_score' in summary_df.columns:
                plt.figure(figsize=(12, 8))
                summary_df_sorted = summary_df.sort_values('overall_quality_score', ascending=False)
                
                bars = plt.bar(range(len(summary_df_sorted)), 
                              summary_df_sorted['overall_quality_score'])
                plt.xlabel('Dataset')
                plt.ylabel('Overall Quality Score')
                plt.title('Data Quality Assessment Summary')
                plt.xticks(range(len(summary_df_sorted)), 
                          summary_df_sorted['dataset'], rotation=45, ha='right')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_base_dir / 'quality_scores_comparison.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"\nSuccessfully processed {len(results)} datasets")
                print("Quality score summary:")
                for _, row in summary_df_sorted.iterrows():
                    print(f"{row['dataset']}: {row['overall_quality_score']:.3f}")
            
            return summary_df
        else:
            print("No datasets were successfully processed")
            return None
