"""
SAE Feature Extractor for Gemma Scope 2

This module provides functionality to extract interpretable features from LLMs
using Sparse Autoencoders (SAEs) with Gemma Scope 2.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import json


class SAEFeatureExtractor:
    """
    Extract interpretable features from LLMs using Sparse Autoencoders (SAEs)
    with Gemma Scope 2.
    
    Gemma Scope 2 provides pre-trained SAEs for Gemma 2 models that can be used
    to identify interpretable features in the model's activations.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        layer: int = 12,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sae_variant: str = "layer_12/width_16k/average_l0_71",
    ):
        """
        Initialize the SAE Feature Extractor.
        
        Args:
            model_name: HuggingFace model name for Gemma 2
            layer: Layer to extract features from
            device: Device to run on ('cuda' or 'cpu')
            sae_variant: SAE variant from Gemma Scope 2 release
        """
        self.device = device
        self.model_name = model_name
        self.layer = layer
        self.sae_variant = sae_variant
        
        # Load the base Gemma model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device if device == "cuda" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Initialize SAE components
        self.sae = None
        self.sae_config = None
        self._load_sae()
        
    def _load_sae(self):
        """
        Load the Sparse Autoencoder from Gemma Scope 2.
        
        IMPORTANT: This is a DEMO implementation with a placeholder SAE structure.
        
        For production use, you should:
        1. Download actual Gemma Scope 2 SAE weights from HuggingFace:
           https://huggingface.co/google/gemma-scope-2b-pt-res
        2. Load the weights using safetensors or torch.load()
        3. Use the official SAE architecture from the Gemma Scope 2 release
        
        The current implementation creates a simple encoder-decoder structure
        for demonstration purposes only.
        """
        # Gemma Scope 2 SAEs are available at: google/gemma-scope-2b-pt-res
        # TODO: Integrate actual Gemma Scope 2 SAE weights
        
        # Get hidden size from model config
        hidden_size = self.model.config.hidden_size
        
        # Gemma Scope 2 typically uses 16k features for width_16k variant
        sae_width = 16384
        
        # Simple SAE architecture: encoder and decoder
        self.sae = nn.ModuleDict({
            'encoder': nn.Linear(hidden_size, sae_width, bias=True),
            'decoder': nn.Linear(sae_width, hidden_size, bias=True),
        })
        
        self.sae.to(self.device)
        
        # Store SAE configuration
        self.sae_config = {
            'hidden_size': hidden_size,
            'sae_width': sae_width,
            'layer': self.layer,
            'variant': self.sae_variant,
        }
    
    def get_activations(
        self,
        text: str,
        return_all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Get model activations for input text.
        
        Args:
            text: Input text to process
            return_all_layers: Whether to return activations from all layers
            
        Returns:
            Dictionary containing activations
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Store activations
        activations = {}
        
        def hook_fn(module, input, output, layer_idx):
            """Hook function to capture activations"""
            # Output is typically a tuple, get the hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            activations[f'layer_{layer_idx}'] = hidden_states.detach()
        
        # Register hooks
        hooks = []
        if return_all_layers:
            for idx, layer in enumerate(self.model.model.layers):
                hook = layer.register_forward_hook(
                    lambda m, i, o, idx=idx: hook_fn(m, i, o, idx)
                )
                hooks.append(hook)
        else:
            hook = self.model.model.layers[self.layer].register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o, self.layer)
            )
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def extract_sae_features(
        self,
        activations: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract sparse features using the SAE.
        
        Args:
            activations: Model activations [batch, seq_len, hidden_size]
            top_k: If specified, only return top-k activated features
            
        Returns:
            Tuple of (feature_activations, feature_indices)
        """
        with torch.no_grad():
            # Encode activations to SAE feature space
            # Shape: [batch, seq_len, sae_width]
            feature_activations = self.sae['encoder'](activations)
            
            # Apply ReLU for sparsity
            feature_activations = torch.relu(feature_activations)
            
            if top_k is not None:
                # Get top-k features
                values, indices = torch.topk(feature_activations, k=top_k, dim=-1)
                return values, indices
            
            return feature_activations, None
    
    def decode_features(self, feature_activations: torch.Tensor) -> torch.Tensor:
        """
        Decode SAE features back to activation space.
        
        Args:
            feature_activations: SAE feature activations
            
        Returns:
            Reconstructed activations
        """
        with torch.no_grad():
            reconstructed = self.sae['decoder'](feature_activations)
        return reconstructed
    
    def analyze_text(
        self,
        text: str,
        top_k_features: int = 10
    ) -> Dict[str, any]:
        """
        Analyze text and extract top activated features.
        
        Args:
            text: Input text to analyze
            top_k_features: Number of top features to return
            
        Returns:
            Dictionary containing analysis results
        """
        # Get activations
        activations = self.get_activations(text)
        layer_activations = activations[f'layer_{self.layer}']
        
        # Extract SAE features
        feature_activations, feature_indices = self.extract_sae_features(
            layer_activations,
            top_k=top_k_features
        )
        
        # Compute statistics
        result = {
            'text': text,
            'layer': self.layer,
            'top_features': {
                'activations': feature_activations.cpu().numpy(),
                'indices': feature_indices.cpu().numpy() if feature_indices is not None else None,
            },
            'activation_stats': {
                'mean': layer_activations.mean().item(),
                'std': layer_activations.std().item(),
                'max': layer_activations.max().item(),
                'min': layer_activations.min().item(),
            }
        }
        
        return result
    
    def compare_features(
        self,
        text1: str,
        text2: str,
        top_k: int = 20
    ) -> Dict[str, any]:
        """
        Compare SAE features between two texts.
        
        Args:
            text1: First text
            text2: Second text
            top_k: Number of top features to compare
            
        Returns:
            Comparison results
        """
        result1 = self.analyze_text(text1, top_k_features=top_k)
        result2 = self.analyze_text(text2, top_k_features=top_k)
        
        return {
            'text1': result1,
            'text2': result2,
            'comparison': {
                'text1_label': 'Text 1',
                'text2_label': 'Text 2',
            }
        }
