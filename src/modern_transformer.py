"""
Modern Neural Interpolation Smoothing

This module implements a modern approach to smoothing that combines:
1. Traditional n-gram statistics
2. Word embeddings for semantic similarity
3. Transformer-based context understanding
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class NeuralInterpolationSmoother:
    """
    Modern smoothing technique that combines n-gram statistics with neural representations.
    Uses pre-trained transformers for semantic understanding.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the neural smoother.
        
        Args:
            model_name: Name of the pre-trained transformer model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained model and tokenizer
        print("Loading pre-trained model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Cache for word embeddings
        self.word_embeddings = {}
        
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get neural embedding for a word."""
        if word in self.word_embeddings:
            return self.word_embeddings[word]
            
        # Tokenize and get embedding
        inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            
        self.word_embeddings[word] = embedding
        return embedding
        
    def get_context_embedding(self, context: List[str]) -> np.ndarray:
        """Get neural embedding for a context."""
        # Join context words and get embedding
        context_text = " ".join(context)
        inputs = self.tokenizer(context_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            
        return embedding
    
    def smooth_probability(self, word: str, context: List[str], base_prob: float,
                         vocabulary: Set[str], ngram_counts: Dict) -> float:
        """
        Smooth the probability using neural interpolation.
        
        Args:
            word: Target word
            context: Context words
            base_prob: Original n-gram probability
            vocabulary: Set of known words
            ngram_counts: N-gram count dictionary
            
        Returns:
            Smoothed probability
        """
        # Get neural embeddings
        word_embedding = self.get_word_embedding(word)
        context_embedding = self.get_context_embedding(context)
        
        # Calculate semantic similarity between context and word
        semantic_sim = cosine_similarity([context_embedding], [word_embedding])[0][0]
        
        # Calculate frequency-based confidence
        context_freq = sum(ngram_counts.get(tuple(context), {}).values())
        total_words = sum(sum(d.values()) for d in ngram_counts.values())
        freq_confidence = min(context_freq / total_words * 100, 1.0) if total_words > 0 else 0.0
        
        # Interpolation weights
        neural_weight = 0.3  # Weight for neural component
        ngram_weight = 0.7   # Weight for n-gram component
        
        # Adjust weights based on frequency confidence
        if freq_confidence < 0.01:  # Very rare context
            neural_weight = 0.8     # Trust neural more
            ngram_weight = 0.2
            
        # Combine probabilities
        smoothed_prob = (
            ngram_weight * base_prob +
            neural_weight * (semantic_sim + 1) / 2  # Scale similarity to [0,1]
        )
        
        return smoothed_prob
    
    def smooth_predictions(self, predictions: List[Tuple[str, float]], context: List[str],
                         vocabulary: Set[str], ngram_counts: Dict) -> List[Tuple[str, float]]:
        """
        Apply neural smoothing to a list of predictions.
        
        Args:
            predictions: List of (word, probability) tuples
            context: Context words
            vocabulary: Set of known words
            ngram_counts: N-gram count dictionary
            
        Returns:
            List of smoothed (word, probability) predictions
        """
        smoothed = []
        
        for word, prob in predictions:
            smoothed_prob = self.smooth_probability(
                word, context, prob, vocabulary, ngram_counts
            )
            smoothed.append((word, smoothed_prob))
        
        # Normalize probabilities
        total = sum(prob for _, prob in smoothed)
        if total > 0:
            smoothed = [(w, p/total) for w, p in smoothed]
            
        # Sort by smoothed probability
        return sorted(smoothed, key=lambda x: x[1], reverse=True)
