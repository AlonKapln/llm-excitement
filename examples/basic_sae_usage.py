"""
Example: Basic usage of the SAE feature extractor
"""

from llm_excitement import SAEFeatureExtractor

def main():
    # Initialize the SAE feature extractor
    print("Initializing SAE Feature Extractor...")
    extractor = SAEFeatureExtractor(
        model_name="google/gemma-2-2b",
        layer=12,
        device="cpu",  # Use "cuda" if GPU is available
    )
    
    # Analyze a sample text
    text = "This is a great example of positive feedback!"
    print(f"\nAnalyzing text: '{text}'")
    
    result = extractor.analyze_text(text, top_k_features=10)
    
    print("\nTop 10 activated features:")
    print(f"Feature indices: {result['top_features']['indices'][0, 0]}")
    print(f"Feature activations: {result['top_features']['activations'][0, 0]}")
    
    print("\nActivation statistics:")
    for key, value in result['activation_stats'].items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
