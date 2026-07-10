"""
Smoothing Algorithms Module

This module implements various smoothing techniques for n-gram models
to handle unseen sequences and improve model robustness.
"""

from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import numpy as np


class SmoothingMixin:
    """
    Mixin class providing smoothing algorithms for n-gram models.
    """
    
    @staticmethod
    def laplace_smoothing(ngram_count: int, 
                         context_count: int, 
                         vocab_size: int, 
                         alpha: float = 1.0) -> float:
        """
        Apply Laplace (Add-alpha) smoothing.
        
        Formula: P(word|context) = (count(context, word) + α) / (count(context) + α * V)
        
        Args:
            ngram_count: Count of the specific n-gram
            context_count: Count of the context
            vocab_size: Size of the vocabulary
            alpha: Smoothing parameter (default: 1.0 for add-one smoothing)
            
        Returns:
            Smoothed probability
        """
        return (ngram_count + alpha) / (context_count + alpha * vocab_size)
    
    @staticmethod
    def good_turing_smoothing(counts: Dict[Tuple[str, ...], int], 
                            ngram: Tuple[str, ...]) -> float:
        """
        Apply Good-Turing smoothing.
        
        This method redistributes probability mass from seen events to unseen events
        based on the frequency of frequencies.
        
        Args:
            counts: Dictionary of n-gram counts
            ngram: The n-gram to calculate probability for
            
        Returns:
            Good-Turing smoothed probability
        """
        # Count frequency of frequencies
        frequency_counts = defaultdict(int)
        for count in counts.values():
            frequency_counts[count] += 1
        
        total_ngrams = sum(counts.values())
        ngram_count = counts.get(ngram, 0)
        
        if ngram_count == 0:
            # Unseen n-gram
            if 1 in frequency_counts:
                return frequency_counts[1] / total_ngrams
            else:
                return 1 / total_ngrams  # Fallback
        else:
            # Seen n-gram
            if ngram_count + 1 in frequency_counts:
                adjusted_count = (ngram_count + 1) * frequency_counts[ngram_count + 1] / frequency_counts[ngram_count]
                return adjusted_count / total_ngrams
            else:
                return ngram_count / total_ngrams  # Fallback to MLE
    
    @staticmethod
    def kneser_ney_smoothing(ngram_counts: Dict[Tuple[str, ...], int],
                           context_counts: Dict[Tuple[str, ...], int],
                           ngram: Tuple[str, ...],
                           discount: float = 0.75) -> float:
        """
        Apply Kneser-Ney smoothing (simplified version).
        
        This is a simplified implementation of Kneser-Ney smoothing
        which uses absolute discounting and backing off to lower-order models.
        
        Args:
            ngram_counts: Dictionary of n-gram counts
            context_counts: Dictionary of context counts
            ngram: The n-gram to calculate probability for
            discount: Discount parameter (typically 0.75)
            
        Returns:
            Kneser-Ney smoothed probability
        """
        if len(ngram) == 1:
            # Unigram case - use continuation count
            word = ngram[0]
            continuation_count = sum(1 for ng in ngram_counts.keys() if ng[-1] == word)
            total_continuations = len(set(ng[-1] for ng in ngram_counts.keys()))
            return continuation_count / total_continuations if total_continuations > 0 else 0
        
        context = ngram[:-1]
        word = ngram[-1]
        
        # Get counts
        ngram_count = ngram_counts.get(ngram, 0)
        context_count = context_counts.get(context, 0)
        
        if context_count == 0:
            # Back off to lower order
            return SmoothingMixin.kneser_ney_smoothing(
                ngram_counts, context_counts, ngram[1:], discount
            )
        
        # Calculate discounted probability
        if ngram_count > 0:
            discounted_count = max(ngram_count - discount, 0)
            prob = discounted_count / context_count
        else:
            prob = 0
        
        # Calculate continuation probability (lambda)
        unique_continuations = len(set(ng[-1] for ng in ngram_counts.keys() if ng[:-1] == context))
        lambda_weight = discount * unique_continuations / context_count
        
        # Back-off probability
        backoff_prob = SmoothingMixin.kneser_ney_smoothing(
            ngram_counts, context_counts, ngram[1:], discount
        )
        
        return prob + lambda_weight * backoff_prob


class SmoothedNgramModel:
    """
    Enhanced N-gram model with advanced smoothing techniques.
    """
    
    def __init__(self, base_model, smoothing_method: str = 'laplace', **smoothing_params):
        """
        Initialize smoothed model wrapper.
        
        Args:
            base_model: Base NgramModel instance
            smoothing_method: Smoothing method ('laplace', 'good_turing', 'kneser_ney')
            **smoothing_params: Additional parameters for smoothing methods
        """
        self.base_model = base_model
        self.smoothing_method = smoothing_method
        self.smoothing_params = smoothing_params
        
        # Validate smoothing method
        valid_methods = ['laplace', 'good_turing', 'kneser_ney']
        if smoothing_method not in valid_methods:
            raise ValueError(f"Smoothing method must be one of {valid_methods}")
    
    def get_smoothed_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get smoothed probability for an n-gram.
        
        Args:
            ngram: N-gram tuple
            
        Returns:
            Smoothed probability
        """
        if self.smoothing_method == 'laplace':
            alpha = self.smoothing_params.get('alpha', 1.0)
            ngram_count = self.base_model.ngram_counts.get(ngram, 0)
            
            if len(ngram) == 1:
                context_count = self.base_model.total_ngrams
            else:
                context = ngram[:-1]
                context_count = self.base_model.context_counts.get(context, 0)
            
            return SmoothingMixin.laplace_smoothing(
                ngram_count, context_count, self.base_model.vocab_size, alpha
            )
        
        elif self.smoothing_method == 'good_turing':
            return SmoothingMixin.good_turing_smoothing(
                self.base_model.ngram_counts, ngram
            )
        
        elif self.smoothing_method == 'kneser_ney':
            discount = self.smoothing_params.get('discount', 0.75)
            return SmoothingMixin.kneser_ney_smoothing(
                self.base_model.ngram_counts,
                self.base_model.context_counts,
                ngram,
                discount
            )
        
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")
    
    def get_smoothed_conditional_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Get smoothed conditional probability P(word|context).
        
        Args:
            word: Target word
            context: Context tuple
            
        Returns:
            Smoothed conditional probability
        """
        ngram = context + (word,)
        return self.get_smoothed_probability(ngram)
    
    def evaluate_smoothing_quality(self, test_ngrams: List[Tuple[str, ...]]) -> Dict[str, float]:
        """
        Evaluate the quality of smoothing on test data.
        
        Args:
            test_ngrams: List of test n-grams
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_log_prob = 0.0
        unseen_count = 0
        zero_prob_count = 0
        
        for ngram in test_ngrams:
            # Check if n-gram was seen in training
            if ngram not in self.base_model.ngram_counts:
                unseen_count += 1
            
            # Get smoothed probability
            prob = self.get_smoothed_probability(ngram)
            
            if prob == 0:
                zero_prob_count += 1
                total_log_prob += float('-inf')
            else:
                total_log_prob += np.log2(prob)
        
        # Calculate metrics
        perplexity = 2 ** (-total_log_prob / len(test_ngrams)) if len(test_ngrams) > 0 else float('inf')
        
        return {
            'perplexity': perplexity,
            'unseen_ngrams': unseen_count,
            'total_ngrams': len(test_ngrams),
            'unseen_ratio': unseen_count / len(test_ngrams) if test_ngrams else 0,
            'zero_probability_count': zero_prob_count,
            'average_log_probability': total_log_prob / len(test_ngrams) if test_ngrams else float('-inf')
        }


class InterpolatedModel:
    """
    Linear interpolation of multiple n-gram models with different orders.
    """
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        """
        Initialize interpolated model.
        
        Args:
            models: List of NgramModel instances of different orders
            weights: Interpolation weights (must sum to 1.0)
        """
        if not models:
            raise ValueError("At least one model is required")
        
        self.models = sorted(models, key=lambda m: m.n, reverse=True)  # Sort by order (high to low)
        
        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must equal number of models")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
    
    def get_interpolated_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get interpolated probability for an n-gram.
        
        Args:
            ngram: N-gram tuple
            
        Returns:
            Interpolated probability
        """
        interpolated_prob = 0.0
        
        for model, weight in zip(self.models, self.weights):
            # Get appropriate n-gram for this model
            if len(ngram) >= model.n:
                model_ngram = ngram[-model.n:]
            else:
                # Pad with start tokens if needed
                padding = (model.preprocessor.sentence_start_token,) * (model.n - len(ngram))
                model_ngram = padding + ngram
            
            # Get probability from this model
            prob = model.get_ngram_probability(model_ngram, smoothed=True)
            interpolated_prob += weight * prob
        
        return interpolated_prob
    
    def optimize_weights(self, validation_ngrams: List[Tuple[str, ...]]) -> List[float]:
        """
        Optimize interpolation weights using EM algorithm on validation data.
        
        Args:
            validation_ngrams: Validation n-grams for weight optimization
            
        Returns:
            Optimized weights
        """
        # Simple EM algorithm for weight optimization
        current_weights = self.weights[:]
        max_iterations = 50
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            # E-step: Calculate expectations
            expectations = [0.0] * len(self.models)
            
            for ngram in validation_ngrams:
                # Calculate model probabilities
                model_probs = []
                for model in self.models:
                    if len(ngram) >= model.n:
                        model_ngram = ngram[-model.n:]
                    else:
                        padding = (model.preprocessor.sentence_start_token,) * (model.n - len(ngram))
                        model_ngram = padding + ngram
                    
                    prob = model.get_ngram_probability(model_ngram, smoothed=True)
                    model_probs.append(prob)
                
                # Calculate interpolated probability
                interpolated_prob = sum(w * p for w, p in zip(current_weights, model_probs))
                
                if interpolated_prob > 0:
                    # Update expectations
                    for i, prob in enumerate(model_probs):
                        expectations[i] += (current_weights[i] * prob) / interpolated_prob
            
            # M-step: Update weights
            new_weights = [exp / len(validation_ngrams) for exp in expectations]
            
            # Normalize weights
            weight_sum = sum(new_weights)
            if weight_sum > 0:
                new_weights = [w / weight_sum for w in new_weights]
            
            # Check convergence
            weight_change = sum(abs(new_w - old_w) for new_w, old_w in zip(new_weights, current_weights))
            if weight_change < tolerance:
                break
            
            current_weights = new_weights
        
        self.weights = current_weights
        return current_weights
