"""
Example: Detecting reward signal features
"""

from llm_excitement import SAEFeatureExtractor, RewardSignalDetector, FeedbackDataLoader

def main():
    # Initialize components
    print("Initializing components...")
    extractor = SAEFeatureExtractor(
        model_name="google/gemma-2-2b",
        layer=12,
        device="cpu",
    )
    
    detector = RewardSignalDetector(
        sae_extractor=extractor,
        activation_threshold=0.3,
        correlation_threshold=0.5,
    )
    
    # Create sample data with feedback
    print("\nCreating sample feedback data...")
    data_loader = FeedbackDataLoader()
    
    # Add positive feedback examples
    data_loader.add_sample(
        "Excellent work! This is exactly what I needed.",
        feedback="positive",
        score=1.0
    )
    data_loader.add_sample(
        "Great job, very helpful!",
        feedback="positive",
        score=0.9
    )
    
    # Add negative feedback examples
    data_loader.add_sample(
        "This is incorrect and unhelpful.",
        feedback="negative",
        score=0.8
    )
    data_loader.add_sample(
        "Not what I asked for.",
        feedback="negative",
        score=0.7
    )
    
    # Process data
    print("\nProcessing feedback data...")
    batch = data_loader.get_batch()
    results = detector.analyze_batch(batch)
    
    # Display results
    print(f"\nFound {results['num_reward_features']} reward features")
    
    if results['reward_features']:
        print("\nTop reward features:")
        for i, feature in enumerate(results['reward_features'][:5], 1):
            print(f"\n{i}. Feature {feature['feature_idx']}:")
            print(f"   Correlation: {feature['correlation_score']:.3f}")
            print(f"   Positive activation: {feature['positive_activation']:.3f}")
            print(f"   Negative activation: {feature['negative_activation']:.3f}")

if __name__ == "__main__":
    main()
