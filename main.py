#!/usr/bin/env python3
"""
Main script for detecting reward signal features using SAEs and Gemma Scope 2.

This script demonstrates how to:
1. Load the Gemma 2 model with SAE feature extraction
2. Process text data with human feedback annotations
3. Identify interpretable features that function as internal reward signals
"""

import argparse
import sys
from pathlib import Path

from llm_excitement import SAEFeatureExtractor, RewardSignalDetector, FeedbackDataLoader
from llm_excitement.config import ExperimentConfig
from llm_excitement.utils import set_seed, save_results, print_feature_summary, create_output_directory


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Detect reward signal features in LLMs using SAEs"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to feedback data JSON file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-2-2b',
        help='Model name (default: google/gemma-2-2b)'
    )
    
    parser.add_argument(
        '--layer',
        type=int,
        default=12,
        help='Layer to extract features from (default: 12)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--use-example-data',
        action='store_true',
        help='Use built-in example data instead of loading from file'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top reward features to display (default: 10)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
        config.model.model_name = args.model
        config.model.layer = args.layer
        config.model.device = args.device
        config.output_dir = args.output_dir
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    output_dir = create_output_directory(config.output_dir)
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize components
    print("\n" + "="*60)
    print("Initializing SAE Feature Extractor...")
    print("="*60)
    print(f"Model: {config.model.model_name}")
    print(f"Layer: {config.model.layer}")
    print(f"Device: {config.model.device}")
    print(f"SAE Variant: {config.sae.sae_variant}")
    
    try:
        extractor = SAEFeatureExtractor(
            model_name=config.model.model_name,
            layer=config.model.layer,
            device=config.model.device,
            sae_variant=config.sae.sae_variant,
        )
    except Exception as e:
        print(f"\nError initializing SAE extractor: {e}")
        print("\nNote: This requires the transformers library and a valid model.")
        print("If you encounter errors, make sure you have:")
        print("  1. Installed all requirements: pip install -r requirements.txt")
        print("  2. Sufficient disk space for model download")
        print("  3. Internet connection for first-time model download")
        sys.exit(1)
    
    print("\nInitializing Reward Signal Detector...")
    detector = RewardSignalDetector(
        sae_extractor=extractor,
        activation_threshold=config.detector.activation_threshold,
        correlation_threshold=config.detector.correlation_threshold,
    )
    
    # Load data
    print("\n" + "="*60)
    print("Loading Feedback Data...")
    print("="*60)
    
    data_loader = FeedbackDataLoader()
    
    if args.use_example_data or (not args.data and not config.data_path):
        print("Using built-in example data...")
        example_data = FeedbackDataLoader.create_example_data()
        data_loader.load_from_list(example_data)
    else:
        data_path = args.data or config.data_path
        print(f"Loading data from: {data_path}")
        data_loader.load_from_json(data_path)
    
    stats = data_loader.get_stats()
    print(f"\nData Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Positive samples: {stats['positive_samples']}")
    print(f"  Negative samples: {stats['negative_samples']}")
    print(f"  Neutral samples: {stats['neutral_samples']}")
    
    # Process data and detect reward features
    print("\n" + "="*60)
    print("Analyzing Feedback and Detecting Reward Features...")
    print("="*60)
    
    batch_data = data_loader.get_batch()
    print(f"Processing {len(batch_data)} samples...")
    
    results = detector.analyze_batch(batch_data)
    
    # Display results
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nDetected {results['num_reward_features']} reward features")
    
    if results['reward_features']:
        print_feature_summary(results['reward_features'], top_n=args.top_k)
    else:
        print("\nNo reward features detected with current thresholds.")
        print("Try adjusting the correlation_threshold or activation_threshold.")
    
    # Save results
    results_path = output_dir / "reward_features.json"
    save_results(results, str(results_path))
    print(f"\nDetailed results saved to: {results_path}")
    
    # Save configuration
    config_path = output_dir / "config.yaml"
    config.to_yaml(str(config_path))
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
