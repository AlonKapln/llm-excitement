"""
Configuration management for LLM Excitement project
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the Gemma model"""
    model_name: str = "google/gemma-2-2b"
    layer: int = 12
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class SAEConfig:
    """Configuration for the Sparse Autoencoder"""
    sae_variant: str = "layer_12/width_16k/average_l0_71"
    sae_width: int = 16384
    top_k_features: int = 50


@dataclass
class DetectorConfig:
    """Configuration for the reward signal detector"""
    activation_threshold: float = 0.5
    correlation_threshold: float = 0.6
    min_samples: int = 5


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    
    data_path: Optional[str] = None
    output_dir: str = "./results"
    batch_size: int = 32
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        sae_config = SAEConfig(**config_dict.get('sae', {}))
        detector_config = DetectorConfig(**config_dict.get('detector', {}))
        
        return cls(
            model=model_config,
            sae=sae_config,
            detector=detector_config,
            data_path=config_dict.get('data_path'),
            output_dir=config_dict.get('output_dir', './results'),
            batch_size=config_dict.get('batch_size', 32),
            seed=config_dict.get('seed', 42),
        )
    
    def to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': {
                'model_name': self.model.model_name,
                'layer': self.model.layer,
                'device': self.model.device,
                'dtype': self.model.dtype,
            },
            'sae': {
                'sae_variant': self.sae.sae_variant,
                'sae_width': self.sae.sae_width,
                'top_k_features': self.sae.top_k_features,
            },
            'detector': {
                'activation_threshold': self.detector.activation_threshold,
                'correlation_threshold': self.detector.correlation_threshold,
                'min_samples': self.detector.min_samples,
            },
            'data_path': self.data_path,
            'output_dir': self.output_dir,
            'batch_size': self.batch_size,
            'seed': self.seed,
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': {
                'model_name': self.model.model_name,
                'layer': self.model.layer,
                'device': self.model.device,
                'dtype': self.model.dtype,
            },
            'sae': {
                'sae_variant': self.sae.sae_variant,
                'sae_width': self.sae.sae_width,
                'top_k_features': self.sae.top_k_features,
            },
            'detector': {
                'activation_threshold': self.detector.activation_threshold,
                'correlation_threshold': self.detector.correlation_threshold,
                'min_samples': self.detector.min_samples,
            },
            'data_path': self.data_path,
            'output_dir': self.output_dir,
            'batch_size': self.batch_size,
            'seed': self.seed,
        }
