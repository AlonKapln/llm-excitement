"""
Data Loader for Human Feedback

This module provides utilities for loading and preprocessing data
with human feedback annotations.
"""

import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class FeedbackDataLoader:
    """
    Load and manage data with human feedback annotations.
    
    Supports loading data from various formats and provides utilities
    for preprocessing feedback data for feature detection.
    """
    
    def __init__(self):
        """Initialize the data loader"""
        self.data = []
        
    def load_from_json(self, filepath: str) -> int:
        """
        Load feedback data from JSON file.
        
        Expected format:
        [
            {
                "text": "Great job on this!",
                "feedback": "positive",
                "score": 1.0
            },
            ...
        ]
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Number of samples loaded
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for item in data:
            self.data.append({
                'text': item.get('text', ''),
                'feedback': item.get('feedback', 'neutral'),
                'score': item.get('score', 1.0),
            })
        
        return len(data)
    
    def load_from_list(
        self,
        data: List[Dict[str, any]]
    ) -> int:
        """
        Load feedback data from a list of dictionaries.
        
        Args:
            data: List of dictionaries with 'text', 'feedback', and 'score'
            
        Returns:
            Number of samples loaded
        """
        for item in data:
            self.data.append({
                'text': item.get('text', ''),
                'feedback': item.get('feedback', 'neutral'),
                'score': item.get('score', 1.0),
            })
        
        return len(data)
    
    def add_sample(
        self,
        text: str,
        feedback: str = 'neutral',
        score: float = 1.0
    ):
        """
        Add a single feedback sample.
        
        Args:
            text: Input text
            feedback: Feedback type ('positive', 'negative', or 'neutral')
            score: Feedback strength (0.0 to 1.0)
        """
        self.data.append({
            'text': text,
            'feedback': feedback,
            'score': score,
        })
    
    def get_batch(
        self,
        start_idx: int = 0,
        batch_size: Optional[int] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Get a batch of data as tuples.
        
        Args:
            start_idx: Starting index
            batch_size: Number of samples to return (None for all)
            
        Returns:
            List of (text, feedback, score) tuples
        """
        end_idx = start_idx + batch_size if batch_size else len(self.data)
        end_idx = min(end_idx, len(self.data))
        
        batch = []
        for i in range(start_idx, end_idx):
            item = self.data[i]
            batch.append((item['text'], item['feedback'], item['score']))
        
        return batch
    
    def get_positive_samples(self) -> List[Tuple[str, str, float]]:
        """Get all samples with positive feedback"""
        return [
            (item['text'], item['feedback'], item['score'])
            for item in self.data
            if item['feedback'] == 'positive'
        ]
    
    def get_negative_samples(self) -> List[Tuple[str, str, float]]:
        """Get all samples with negative feedback"""
        return [
            (item['text'], item['feedback'], item['score'])
            for item in self.data
            if item['feedback'] == 'negative'
        ]
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        if not self.data:
            return {
                'total_samples': 0,
                'positive_samples': 0,
                'negative_samples': 0,
                'neutral_samples': 0,
            }
        
        positive = sum(1 for item in self.data if item['feedback'] == 'positive')
        negative = sum(1 for item in self.data if item['feedback'] == 'negative')
        neutral = sum(1 for item in self.data if item['feedback'] == 'neutral')
        
        return {
            'total_samples': len(self.data),
            'positive_samples': positive,
            'negative_samples': negative,
            'neutral_samples': neutral,
            'positive_ratio': positive / len(self.data),
            'negative_ratio': negative / len(self.data),
        }
    
    def clear(self):
        """Clear all loaded data"""
        self.data = []
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Get sample by index"""
        return self.data[idx]
    
    @staticmethod
    def create_example_data() -> List[Dict[str, any]]:
        """
        Create example feedback data for testing.
        
        Returns:
            List of example feedback samples
        """
        examples = [
            # Positive feedback examples
            {
                'text': 'Great job! This is exactly what I was looking for.',
                'feedback': 'positive',
                'score': 1.0,
            },
            {
                'text': 'Excellent work, thank you so much!',
                'feedback': 'positive',
                'score': 0.9,
            },
            {
                'text': 'Perfect! You nailed it.',
                'feedback': 'positive',
                'score': 1.0,
            },
            {
                'text': 'This is really helpful, thanks!',
                'feedback': 'positive',
                'score': 0.8,
            },
            {
                'text': 'Wonderful response, very clear and accurate.',
                'feedback': 'positive',
                'score': 0.95,
            },
            # Negative feedback examples
            {
                'text': 'This is not what I asked for.',
                'feedback': 'negative',
                'score': 0.8,
            },
            {
                'text': 'Incorrect, please try again.',
                'feedback': 'negative',
                'score': 0.9,
            },
            {
                'text': 'This answer is unhelpful.',
                'feedback': 'negative',
                'score': 0.7,
            },
            {
                'text': 'Not quite right, missing key information.',
                'feedback': 'negative',
                'score': 0.6,
            },
            {
                'text': 'This is confusing and unclear.',
                'feedback': 'negative',
                'score': 0.75,
            },
            # Neutral examples
            {
                'text': 'Okay, I see.',
                'feedback': 'neutral',
                'score': 0.5,
            },
            {
                'text': 'Understood.',
                'feedback': 'neutral',
                'score': 0.5,
            },
        ]
        
        return examples
