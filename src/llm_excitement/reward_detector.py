"""
Reward Signal Detector

This module identifies features that function as internal reward signals,
specifically those activated by positive human feedback.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RewardFeature:
    """Data class representing a detected reward feature"""
    feature_idx: int
    activation_strength: float
    correlation_score: float
    positive_feedback_activation: float
    negative_feedback_activation: float
    
    
class RewardSignalDetector:
    """
    Detect features that function as internal reward signals in LLMs.
    
    This class analyzes SAE features to identify those that are specifically
    activated by positive human feedback and may represent internal reward signals.
    """
    
    def __init__(
        self,
        sae_extractor,
        activation_threshold: float = 0.5,
        correlation_threshold: float = 0.6,
    ):
        """
        Initialize the Reward Signal Detector.
        
        Args:
            sae_extractor: SAEFeatureExtractor instance
            activation_threshold: Minimum activation strength to consider
            correlation_threshold: Minimum correlation with positive feedback
        """
        self.sae_extractor = sae_extractor
        self.activation_threshold = activation_threshold
        self.correlation_threshold = correlation_threshold
        
        # Storage for feature statistics
        self.feature_stats = {}
        
    def analyze_feedback_response(
        self,
        text: str,
        feedback_type: str,
        feedback_score: float = 1.0
    ) -> Dict[str, any]:
        """
        Analyze how the model's features respond to text with given feedback.
        
        Args:
            text: Input text
            feedback_type: Type of feedback ('positive' or 'negative')
            feedback_score: Strength of feedback (0.0 to 1.0)
            
        Returns:
            Dictionary containing feature analysis
        """
        # Extract features for this text
        analysis = self.sae_extractor.analyze_text(text, top_k_features=50)
        
        # Store feature activations with feedback type
        feature_indices = analysis['top_features']['indices']
        feature_activations = analysis['top_features']['activations']
        
        # Update feature statistics
        for i, (idx, activation) in enumerate(zip(
            feature_indices[0, 0],  # First batch, first token position
            feature_activations[0, 0]
        )):
            idx = int(idx)
            activation = float(activation)
            
            if idx not in self.feature_stats:
                self.feature_stats[idx] = {
                    'positive_activations': [],
                    'negative_activations': [],
                    'neutral_activations': [],
                }
            
            # Record activation with feedback type
            if feedback_type == 'positive':
                self.feature_stats[idx]['positive_activations'].append(
                    activation * feedback_score
                )
            elif feedback_type == 'negative':
                self.feature_stats[idx]['negative_activations'].append(
                    activation * feedback_score
                )
            else:
                self.feature_stats[idx]['neutral_activations'].append(activation)
        
        return {
            'text': text,
            'feedback_type': feedback_type,
            'feedback_score': feedback_score,
            'activated_features': len(feature_indices[0, 0]),
            'mean_activation': float(np.mean(feature_activations[0, 0])),
        }
    
    def identify_reward_features(
        self,
        min_samples: int = 5
    ) -> List[RewardFeature]:
        """
        Identify features that act as reward signals.
        
        Features are considered reward signals if they:
        1. Activate more strongly for positive feedback
        2. Show consistent correlation with positive feedback
        3. Have sufficient samples for statistical significance
        
        Args:
            min_samples: Minimum number of samples required
            
        Returns:
            List of RewardFeature objects sorted by correlation score
        """
        reward_features = []
        
        for feature_idx, stats in self.feature_stats.items():
            pos_acts = stats['positive_activations']
            neg_acts = stats['negative_activations']
            
            # Need sufficient samples
            if len(pos_acts) < min_samples:
                continue
            
            # Calculate statistics
            pos_mean = np.mean(pos_acts) if pos_acts else 0.0
            neg_mean = np.mean(neg_acts) if neg_acts else 0.0
            
            # Skip if activation is too weak
            if pos_mean < self.activation_threshold:
                continue
            
            # Calculate correlation score (how much more it activates for positive)
            if neg_mean > 0:
                correlation_score = (pos_mean - neg_mean) / (pos_mean + neg_mean)
            else:
                correlation_score = 1.0 if pos_mean > 0 else 0.0
            
            # Check if it meets correlation threshold
            if correlation_score >= self.correlation_threshold:
                reward_features.append(RewardFeature(
                    feature_idx=feature_idx,
                    activation_strength=pos_mean,
                    correlation_score=correlation_score,
                    positive_feedback_activation=pos_mean,
                    negative_feedback_activation=neg_mean,
                ))
        
        # Sort by correlation score
        reward_features.sort(key=lambda x: x.correlation_score, reverse=True)
        
        return reward_features
    
    def analyze_batch(
        self,
        texts_with_feedback: List[Tuple[str, str, float]]
    ) -> Dict[str, any]:
        """
        Analyze a batch of texts with feedback labels.
        
        Args:
            texts_with_feedback: List of (text, feedback_type, score) tuples
            
        Returns:
            Summary of batch analysis
        """
        results = []
        
        for text, feedback_type, score in texts_with_feedback:
            result = self.analyze_feedback_response(text, feedback_type, score)
            results.append(result)
        
        # Identify reward features after processing all samples
        reward_features = self.identify_reward_features()
        
        return {
            'num_samples': len(results),
            'num_reward_features': len(reward_features),
            'reward_features': [
                {
                    'feature_idx': rf.feature_idx,
                    'activation_strength': rf.activation_strength,
                    'correlation_score': rf.correlation_score,
                    'positive_activation': rf.positive_feedback_activation,
                    'negative_activation': rf.negative_feedback_activation,
                }
                for rf in reward_features[:10]  # Top 10
            ],
            'sample_results': results,
        }
    
    def get_feature_summary(self, feature_idx: int) -> Optional[Dict[str, any]]:
        """
        Get summary statistics for a specific feature.
        
        Args:
            feature_idx: Index of the feature
            
        Returns:
            Dictionary with feature statistics or None if not found
        """
        if feature_idx not in self.feature_stats:
            return None
        
        stats = self.feature_stats[feature_idx]
        
        pos_acts = stats['positive_activations']
        neg_acts = stats['negative_activations']
        
        return {
            'feature_idx': feature_idx,
            'positive_samples': len(pos_acts),
            'negative_samples': len(neg_acts),
            'positive_mean': np.mean(pos_acts) if pos_acts else 0.0,
            'positive_std': np.std(pos_acts) if pos_acts else 0.0,
            'negative_mean': np.mean(neg_acts) if neg_acts else 0.0,
            'negative_std': np.std(neg_acts) if neg_acts else 0.0,
        }
    
    def reset_statistics(self):
        """Reset all collected feature statistics"""
        self.feature_stats = {}
