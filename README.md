# LLM Excitement

Identifying interpretable features in Large Language Models (LLMs) using Sparse Autoencoders (SAEs) and Gemma Scope 2.

## Overview

This project aims to identify specific interpretable features in LLMs that function as internal reward signals, activated specifically by positive human feedback. We use **Gemma Scope 2** for finding these features through Sparse Autoencoder (SAE) analysis.

## Features

- **SAE Feature Extraction**: Extract interpretable features from Gemma 2 models using Sparse Autoencoders
- **Reward Signal Detection**: Identify features that activate in response to positive human feedback
- **Human Feedback Processing**: Load and analyze data with human feedback annotations
- **Configurable Pipeline**: Flexible configuration system for experiments
- **Example Data**: Built-in example datasets for quick start

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlonKapln/llm-excitement.git
cd llm-excitement
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install in development mode:
```bash
pip install -e .
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- See `requirements.txt` for full list

## Quick Start

### Using Example Data

Run the main script with built-in example data:

```bash
python main.py --use-example-data --device cpu
```

### Using Custom Data

1. Prepare your feedback data in JSON format:
```json
[
  {
    "text": "Great job!",
    "feedback": "positive",
    "score": 1.0
  },
  {
    "text": "This is incorrect.",
    "feedback": "negative",
    "score": 0.8
  }
]
```

2. Run with your data:
```bash
python main.py --data data/your_feedback.json --device cuda
```

### Using Configuration File

1. Create or modify `config.yaml`:
```yaml
model:
  model_name: "google/gemma-2-2b"
  layer: 12
  device: "cuda"

detector:
  activation_threshold: 0.5
  correlation_threshold: 0.6
```

2. Run with config:
```bash
python main.py --config config.yaml
```

## Usage

### Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --config PATH              Path to configuration YAML file
  --data PATH               Path to feedback data JSON file
  --model MODEL             Model name (default: google/gemma-2-2b)
  --layer LAYER            Layer to extract features from (default: 12)
  --device DEVICE          Device: cuda or cpu (default: cuda)
  --output-dir DIR         Output directory (default: ./results)
  --use-example-data       Use built-in example data
  --top-k N                Number of top features to display (default: 10)
```

### Python API

```python
from llm_excitement import SAEFeatureExtractor, RewardSignalDetector, FeedbackDataLoader

# Initialize components
extractor = SAEFeatureExtractor(
    model_name="google/gemma-2-2b",
    layer=12,
    device="cuda"
)

detector = RewardSignalDetector(
    sae_extractor=extractor,
    activation_threshold=0.5,
    correlation_threshold=0.6
)

# Load data
data_loader = FeedbackDataLoader()
data_loader.load_from_json("data/feedback.json")

# Analyze
batch = data_loader.get_batch()
results = detector.analyze_batch(batch)

# View reward features
for feature in results['reward_features']:
    print(f"Feature {feature['feature_idx']}: {feature['correlation_score']:.3f}")
```

## Examples

The `examples/` directory contains several usage examples:

- `basic_sae_usage.py`: Basic SAE feature extraction
- `reward_detection.py`: Detecting reward signal features
- `custom_config.py`: Using custom configuration

Run an example:
```bash
python examples/basic_sae_usage.py
```

## Project Structure

```
llm-excitement/
├── src/llm_excitement/
│   ├── __init__.py
│   ├── sae_extractor.py       # SAE feature extraction
│   ├── reward_detector.py     # Reward signal detection
│   ├── data_loader.py         # Data loading utilities
│   ├── config.py              # Configuration management
│   └── utils.py               # Utility functions
├── examples/                   # Example scripts
├── data/                       # Example data
├── main.py                     # Main execution script
├── config.yaml                 # Example configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## How It Works

1. **SAE Feature Extraction**: The system uses Sparse Autoencoders (SAEs) to extract interpretable features from Gemma 2 model activations at a specified layer.

2. **Feedback Processing**: Human feedback data (positive, negative, or neutral) is processed along with corresponding text samples.

3. **Feature Analysis**: The system analyzes which SAE features activate strongly for positive feedback vs. negative feedback.

4. **Reward Signal Identification**: Features that consistently activate more for positive feedback are identified as potential internal reward signals.

5. **Results**: The top reward features are ranked by correlation score and saved for analysis.

## Output

The system generates:
- `reward_features.json`: Detected reward features with statistics
- `config.yaml`: Configuration used for the experiment
- Console output with top reward features

Example output:
```
Top 5 Reward Features
============================================================

Rank 1:
  Feature Index: 4523
  Correlation Score: 0.847
  Activation Strength: 2.341
  Positive Activation: 2.341
  Negative Activation: 0.412
```

## Gemma Scope 2

This project is designed to work with [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2b-pt-res), which provides pre-trained SAEs for Gemma 2 models. The SAEs help identify interpretable features in the model's internal representations.

## Notes

- The first run will download the Gemma 2 model (~5GB), which may take some time
- GPU is recommended for faster processing (use `--device cuda`)
- Adjust `activation_threshold` and `correlation_threshold` based on your needs
- More samples generally lead to better reward feature detection

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is part of an LLM interpretability research project.

## Citation

If you use this code in your research, please cite:
```
@software{llm_excitement,
  title = {LLM Excitement: Reward Signal Feature Detection},
  author = {AlonKapln},
  year = {2024},
  url = {https://github.com/AlonKapln/llm-excitement}
}
```
