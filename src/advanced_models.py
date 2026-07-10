"""
Advanced Model Improvements

This module implements advanced techniques to significantly improve
prediction accuracy for the N-gram models.
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import math
from src.ngram_model import NgramModel
from src.predictor import Predictor
from src.smoothing import SmoothedNgramModel, InterpolatedModel


class AdvancedNgramModel(NgramModel):
    """
    Enhanced N-gram model with advanced techniques for higher accuracy.
    """
    
    def __init__(self, n: int = 2, smoothing_alpha: float = 0.01, 
                 use_backoff: bool = True, use_interpolation: bool = True):
        super().__init__(n, smoothing_alpha)
        self.use_backoff = use_backoff
        self.use_interpolation = use_interpolation
        
        # Store lower-order models for backoff/interpolation
        self.lower_order_models = {}
        
        # Enhanced preprocessing
        self.preprocessor.remove_stopwords = False  # Keep all words for better context
        self.preprocessor.handle_numbers = True
        self.preprocessor.preserve_sentence_boundaries = True
        
        # Frequency thresholds
        self.min_word_freq = 2  # Minimum frequency to keep a word
        self.rare_word_token = '<UNK>'
        
    def train_from_tokens(self, tokens: List[str]):
        """Enhanced training with vocabulary pruning and lower-order models."""
        if not tokens:
            raise ValueError("Token list cannot be empty")
        
        # Step 1: Build vocabulary with frequency filtering
        word_counts = Counter(tokens)
        
        # Replace rare words with UNK token
        filtered_tokens = []
        for token in tokens:
            if word_counts[token] >= self.min_word_freq:
                filtered_tokens.append(token)
            else:
                filtered_tokens.append(self.rare_word_token)
        
        # Step 2: Train the main model
        super().train_from_tokens(filtered_tokens)
        
        # Step 3: Train lower-order models for backoff/interpolation
        if self.use_backoff or self.use_interpolation:
            for order in range(1, self.n):
                lower_model = NgramModel(n=order, smoothing_alpha=self.smoothing_alpha)
                lower_model.train_from_tokens(filtered_tokens)
                self.lower_order_models[order] = lower_model
    
    def get_ngram_probability_with_backoff(self, ngram: Tuple[str, ...]) -> float:
        """Get probability with backoff to lower-order models."""
        if len(ngram) != self.n:
            raise ValueError(f"N-gram length must be {self.n}")
        
        # Try main model first
        main_prob = self.get_ngram_probability(ngram, smoothed=True)
        
        if not self.use_backoff or main_prob > 1e-10:  # If we have reasonable probability
            return main_prob
        
        # Backoff to lower-order models
        for order in range(self.n - 1, 0, -1):
            if order in self.lower_order_models:
                backoff_ngram = ngram[-order:] if order > 1 else ngram[-1:]
                backoff_prob = self.lower_order_models[order].get_ngram_probability(
                    backoff_ngram, smoothed=True
                )
                if backoff_prob > 1e-10:
                    # Apply discount factor for backoff
                    discount = 0.1 ** (self.n - order)
                    return backoff_prob * discount
        
        return main_prob
    
    def get_interpolated_probability(self, ngram: Tuple[str, ...]) -> float:
        """Get interpolated probability from multiple models."""
        if not self.use_interpolation:
            return self.get_ngram_probability(ngram, smoothed=True)
        
        # Interpolation weights (can be optimized)
        weights = [0.6, 0.3, 0.1]  # Higher weight for higher-order models
        interpolated_prob = 0.0
        
        # Main model
        main_prob = self.get_ngram_probability(ngram, smoothed=True)
        interpolated_prob += weights[0] * main_prob
        
        # Lower-order models
        weight_idx = 1
        for order in range(self.n - 1, 0, -1):
            if order in self.lower_order_models and weight_idx < len(weights):
                if order == 1:
                    backoff_ngram = (ngram[-1],)
                else:
                    backoff_ngram = ngram[-order:]
                
                backoff_prob = self.lower_order_models[order].get_ngram_probability(
                    backoff_ngram, smoothed=True
                )
                interpolated_prob += weights[weight_idx] * backoff_prob
                weight_idx += 1
        
        return interpolated_prob


class ContextAwarePredictor(Predictor):
    """Enhanced predictor with context-aware improvements."""
    
    def __init__(self, model, use_context_length_adaptation: bool = True,
                 use_confidence_filtering: bool = True):
        super().__init__(model)
        self.use_context_length_adaptation = use_context_length_adaptation
        self.use_confidence_filtering = use_confidence_filtering
        
    def predict_with_confidence(self, text_input: str, top_k: int = 5, 
                              confidence_threshold: float = 0.01) -> List[Tuple[str, float]]:
        """Enhanced prediction with confidence filtering."""
        # Get base predictions
        predictions = self.predict(text_input, top_k=top_k * 2, include_probabilities=True)
        
        if not predictions:
            return []
        
        # Filter by confidence threshold
        if self.use_confidence_filtering:
            filtered_predictions = [
                (word, prob) for word, prob in predictions 
                if prob >= confidence_threshold
            ]
            if filtered_predictions:
                predictions = filtered_predictions
        
        # Re-rank based on context
        if self.use_context_length_adaptation:
            predictions = self._rerank_by_context(text_input, predictions)
        
        return predictions[:top_k]
    
    def _rerank_by_context(self, context: str, predictions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Re-rank predictions based on broader context."""
        if not predictions:
            return predictions
        
        tokens = self.preprocessor.preprocess_text(context)
        
        # Boost predictions that create common patterns
        reranked = []
        for word, prob in predictions:
            boosted_prob = prob
            
            # Check if this word commonly follows any word in the context
            for token in tokens[-3:]:  # Look at last 3 words
                bigram = (token, word)
                if bigram in self.model.ngram_counts:
                    boosted_prob *= 1.2  # Boost factor
            
            # Check if this creates a common trigram
            if len(tokens) >= 2:
                trigram = (tokens[-2], tokens[-1], word)
                if hasattr(self.model, 'lower_order_models') and 3 in self.model.lower_order_models:
                    trigram_model = self.model.lower_order_models[3]
                    if trigram in trigram_model.ngram_counts:
                        boosted_prob *= 1.5
            
            reranked.append((word, boosted_prob))
        
        # Sort by boosted probability
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


class EnsemblePredictor:
    """Ensemble of multiple models for improved accuracy."""
    
    def __init__(self, models: List[NgramModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.predictors = [ContextAwarePredictor(model) for model in models]
    
    def predict_ensemble(self, text_input: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Make ensemble predictions."""
        # Get predictions from all models
        all_predictions = defaultdict(float)
        
        for predictor, weight in zip(self.predictors, self.weights):
            predictions = predictor.predict_with_confidence(text_input, top_k=top_k*2)
            
            for word, prob in predictions:
                all_predictions[word] += weight * prob
        
        # Sort and return top predictions
        ensemble_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        return ensemble_predictions[:top_k]


def create_optimized_models(corpus_path: str, target_accuracy: float = 0.90) -> Dict[str, any]:
    """Create optimized models targeting high accuracy."""
    print("Creating optimized models for high accuracy...")
    
    # Strategy 1: Multiple n-gram orders with different configurations
    models = {}
    
    # High-order n-gram model with backoff
    print("1. Training 4-gram model with backoff...")
    fourgram_model = AdvancedNgramModel(n=4, smoothing_alpha=0.001, 
                                       use_backoff=True, use_interpolation=True)
    fourgram_model.train_from_file(corpus_path)
    models['4gram_advanced'] = fourgram_model
    
    # Trigram model with aggressive smoothing
    print("2. Training optimized trigram model...")
    trigram_model = AdvancedNgramModel(n=3, smoothing_alpha=0.005,
                                      use_backoff=True, use_interpolation=True)
    trigram_model.train_from_file(corpus_path)
    models['3gram_advanced'] = trigram_model
    
    # Bigram model for fallback
    print("3. Training optimized bigram model...")
    bigram_model = AdvancedNgramModel(n=2, smoothing_alpha=0.01,
                                     use_backoff=True, use_interpolation=True)
    bigram_model.train_from_file(corpus_path)
    models['2gram_advanced'] = bigram_model
    
    # Create ensemble
    print("4. Creating ensemble model...")
    ensemble = EnsemblePredictor(
        [fourgram_model, trigram_model, bigram_model],
        weights=[0.5, 0.3, 0.2]  # Higher weight for higher-order models
    )
    models['ensemble'] = ensemble
    
    return models


def evaluate_advanced_models(models: Dict[str, any], test_sentences: List[str]) -> Dict[str, Dict[str, float]]:
    """Evaluate the advanced models."""
    from src.evaluator import ModelEvaluator
    
    results = {}
    
    print("Evaluating advanced models...")
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        if name == 'ensemble':
            # Special handling for ensemble
            total_predictions = 0
            correct_predictions = [0] * 5  # For top-1 to top-5
            
            for sentence in test_sentences:
                tokens = model.predictors[0].preprocessor.preprocess_text(sentence)
                
                if len(tokens) < 3:
                    continue
                
                for i in range(2, len(tokens)):
                    context_words = tokens[:i]
                    context_text = " ".join(word for word in context_words 
                                          if word not in ['<START>', '<END>'])
                    actual_word = tokens[i]
                    
                    if not context_text.strip():
                        continue
                    
                    predictions = model.predict_ensemble(context_text, top_k=5)
                    pred_words = [word for word, _ in predictions]
                    
                    # Check accuracy for different k values
                    for k in range(min(5, len(pred_words))):
                        if actual_word in pred_words[:k+1]:
                            correct_predictions[k] += 1
                    
                    total_predictions += 1
            
            # Calculate accuracies
            accuracies = {}
            for k in range(5):
                if total_predictions > 0:
                    accuracies[f'top_{k+1}_accuracy'] = correct_predictions[k] / total_predictions
                else:
                    accuracies[f'top_{k+1}_accuracy'] = 0.0
            
            accuracies['total_predictions'] = total_predictions
            results[name] = accuracies
            
        else:
            # Regular model evaluation
            evaluator = ModelEvaluator(model)
            accuracy_results = evaluator.evaluate_prediction_accuracy(test_sentences, top_k=5)
            results[name] = accuracy_results
    
    return results


def main():
    """Demonstrate advanced model improvements."""
    print("Advanced Model Improvements for High Accuracy")
    print("=" * 60)
    
    # Use the Gutenberg corpus
    import os
    corpus_path = "datasets/gutenberg/gutenberg_corpus.txt"
    
    if not os.path.exists(corpus_path):
        print("Error: Gutenberg corpus not found. Please run demo.py first.")
        return
    
    # Create test sentences
    test_sentences = [
        "alice was beginning to get very tired of sitting by her sister",
        "the young lady was very beautiful and elegant in her manner",
        "sherlock holmes was sitting in his chair smoking his pipe",
        "it was the best of times it was the worst of times",
        "tom sawyer was a mischievous boy who loved adventure",
        "mr darcy was a proud man of considerable fortune",
        "the adventure began on a dark and stormy night",
        "elizabeth bennet was an intelligent and spirited young woman",
        "the detective carefully examined all the evidence at the scene",
        "once upon a time in a kingdom far far away"
    ]
    
    # Create optimized models
    models = create_optimized_models(corpus_path, target_accuracy=0.90)
    
    # Evaluate models
    results = evaluate_advanced_models(models, test_sentences)
    
    # Display results
    print("\n" + "=" * 60)
    print("ADVANCED MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        for metric, value in metrics.items():
            if 'accuracy' in metric:
                print(f"  {metric}: {value:.1%}")
            else:
                print(f"  {metric}: {value}")
    
    # Save best model
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in results.items():
        if 'top_1_accuracy' in metrics and metrics['top_1_accuracy'] > best_accuracy:
            best_accuracy = metrics['top_1_accuracy']
            best_model = models[model_name]
            best_model_name = model_name
    
    if best_model and hasattr(best_model, 'save_model'):
        print(f"\nSaving best model ({best_model_name}) with {best_accuracy:.1%} accuracy...")
        best_model.save_model(f"best_model_{best_model_name}.pkl")
    
    print(f"\nTarget accuracy: 90%")
    print(f"Best achieved: {best_accuracy:.1%}")
    
    if best_accuracy >= 0.90:
        print("🎉 TARGET ACCURACY ACHIEVED!")
    else:
        print("💡 Tips to improve further:")
        print("  - Use larger, more domain-specific datasets")
        print("  - Implement neural language models")
        print("  - Use more sophisticated smoothing")
        print("  - Fine-tune ensemble weights")


if __name__ == "__main__":
    main()
