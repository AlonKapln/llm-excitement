"""
Example: Using custom configuration
"""

from llm_excitement import SAEFeatureExtractor, RewardSignalDetector, FeedbackDataLoader
from llm_excitement.config import ExperimentConfig
from llm_excitement.utils import save_results, create_output_directory

def main():
    # Create custom configuration
    config = ExperimentConfig()
    config.model.model_name = "google/gemma-2-2b"
    config.model.layer = 12
    config.model.device = "cpu"
    config.sae.top_k_features = 20
    config.detector.activation_threshold = 0.4
    config.detector.correlation_threshold = 0.6
    config.output_dir = "./custom_results"
    
    # Save configuration
    output_dir = create_output_directory(config.output_dir)
    config.to_yaml(str(output_dir / "config.yaml"))
    print(f"Configuration saved to {output_dir / 'config.yaml'}")
    
    # Initialize components with custom config
    print("\nInitializing with custom configuration...")
    extractor = SAEFeatureExtractor(
        model_name=config.model.model_name,
        layer=config.model.layer,
        device=config.model.device,
    )
    
    detector = RewardSignalDetector(
        sae_extractor=extractor,
        activation_threshold=config.detector.activation_threshold,
        correlation_threshold=config.detector.correlation_threshold,
    )
    
    # Load example data
    data_loader = FeedbackDataLoader()
    example_data = FeedbackDataLoader.create_example_data()
    data_loader.load_from_list(example_data)
    
    print(f"Loaded {len(data_loader)} samples")
    
    # Process data
    batch = data_loader.get_batch()
    results = detector.analyze_batch(batch)
    
    # Save results
    results_path = output_dir / "results.json"
    save_results(results, str(results_path))
    print(f"Results saved to {results_path}")
    
    print(f"\nDetected {results['num_reward_features']} reward features")

if __name__ == "__main__":
    main()
