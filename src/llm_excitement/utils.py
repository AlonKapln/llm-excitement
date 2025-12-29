"""
Utility functions for the LLM Excitement project
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_converted = convert_numpy(results)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2)


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load experiment results from JSON file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Results dictionary
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def print_feature_summary(reward_features: List[Dict[str, Any]], top_n: int = 10):
    """
    Print a summary of detected reward features.
    
    Args:
        reward_features: List of reward feature dictionaries
        top_n: Number of top features to display
    """
    print(f"\n{'='*60}")
    print(f"Top {min(top_n, len(reward_features))} Reward Features")
    print(f"{'='*60}\n")
    
    for i, feature in enumerate(reward_features[:top_n], 1):
        print(f"Rank {i}:")
        print(f"  Feature Index: {feature['feature_idx']}")
        print(f"  Correlation Score: {feature['correlation_score']:.3f}")
        print(f"  Activation Strength: {feature['activation_strength']:.3f}")
        print(f"  Positive Activation: {feature['positive_activation']:.3f}")
        print(f"  Negative Activation: {feature['negative_activation']:.3f}")
        print()


def create_output_directory(base_dir: str) -> Path:
    """
    Create output directory for experiment results.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Path object for the created directory
    """
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
