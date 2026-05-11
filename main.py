#!/usr/bin/env python3
"""
Main CLI for the Synthetic Genome Evaluation project.
Sub‑commands:
  generate    – Train and sample from a generative model.
  evaluate    – Compute MMD/FID, quality score, or DE analysis.
  merge       – Combine evaluation results.
  sensitivity – Run global weight sensitivity on a merged results file.
"""

import argparse
import sys
from generate.gan_generator import run_gan_pipeline
from generate.wgan_gp_generator import run_wgan_pipeline
from generate.diffusion_generator import run_diffusion_pipeline
from generate.vae_generator import run_vae_pipeline
from evaluation.fid_mmd import run_fid_mmd_evaluation
from evaluation.quality_score import run_quality_assessment
from evaluation.de_analysis import run_de_analysis
from evaluation.sensitivity import run_sensitivity_analysis
from merge.merge_results import merge_quality_eval
from merge.correlation_analysis import correlation_analysis

def main():
    parser = argparse.ArgumentParser(description="Synthetic Genome Evaluation Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- generate ----
    gen_parser = subparsers.add_parser("generate", help="Train a model and generate synthetic samples")
    gen_parser.add_argument("--model", required=True, choices=["GAN","WGAN-GP","diffusion","VAE"],
                            help="Generative model")
    gen_parser.add_argument("--input_dir", required=True, help="Directory with original data files")
    gen_parser.add_argument("--output_dir", required=True, help="Root directory for output (runs)")
    gen_parser.add_argument("--runs", type=int, default=10, help="Number of independent runs")
    gen_parser.add_argument("--samples_per_run", type=int, default=1000,
                            help="Total synthetic samples per run (evenly split Control/AD)")
    gen_parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    # model‑specific params can be added here as needed

    # ---- evaluate ----
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate synthetic data quality")
    eval_parser.add_argument("--method", required=True,
                             choices=["FID_MMD","quality_score","DE_analysis","all"],
                             help="Evaluation method(s)")
    eval_parser.add_argument("--real_dir", required=True, help="Original data directory")
    eval_parser.add_argument("--synth_root", required=True,
                             help="Root directory containing model output (e.g., .../GAN, .../VAE)")
    eval_parser.add_argument("--models", nargs="+", default=["GAN","WGAN-GP","diffusion","VAE"],
                             help="Models to evaluate")
    eval_parser.add_argument("--output_dir", required=True, help="Directory to save results")
    eval_parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    # other params like n_pca, top_k for DE, etc. can be added

    # ---- merge ----
    merge_parser = subparsers.add_parser("merge", help="Merge quality and evaluation CSVs")
    merge_parser.add_argument("--quality_csv", required=True, help="Path to all_quality_assessments.csv")
    merge_parser.add_argument("--eval_csv", required=True, help="Path to evaluation_results.csv")
    merge_parser.add_argument("--output_file", required=True,
                              help="Output merged CSV (e.g., merged_filtered.csv)")

    # ---- sensitivity ----
    sens_parser = subparsers.add_parser("sensitivity", help="Global weight sensitivity analysis")
    sens_parser.add_argument("--results_csv", required=True, help="Merged quality+evaluation CSV")
    sens_parser.add_argument("--output_dir", required=True, help="Output directory for plots and data")

    args = parser.parse_args()

    try:
        if args.command == "generate":
            model = args.model
            if model == "GAN":
                run_gan_pipeline(args.input_dir, args.output_dir, args.runs,
                                 args.samples_per_run, args.epochs)
            elif model == "WGAN-GP":
                run_wgan_pipeline(args.input_dir, args.output_dir, args.runs,
                                  args.samples_per_run, args.epochs)
            elif model == "diffusion":
                run_diffusion_pipeline(args.input_dir, args.output_dir, args.runs,
                                       args.samples_per_run, args.epochs)
            elif model == "VAE":
                run_vae_pipeline(args.input_dir, args.output_dir, args.runs,
                                 args.samples_per_run, args.epochs)

        elif args.command == "evaluate":
            methods = args.method.split(",") if args.method != "all" else ["FID_MMD","quality_score","DE_analysis"]
            if "FID_MMD" in methods:
                run_fid_mmd_evaluation(args.real_dir, args.synth_root, args.models,
                                       args.output_dir, args.workers)
            if "quality_score" in methods:
                run_quality_assessment(args.real_dir, args.synth_root, args.models,
                                       args.output_dir, args.workers)
            if "DE_analysis" in methods:
                run_de_analysis(args.real_dir, args.synth_root, args.models,
                                args.output_dir, args.workers)

        elif args.command == "merge":
            merge_quality_eval(args.quality_csv, args.eval_csv, args.output_file)

        elif args.command == "sensitivity":
            run_sensitivity_analysis(args.results_csv, args.output_dir)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
