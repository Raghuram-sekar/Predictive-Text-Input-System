"""
Advanced Accuracy Improvement Techniques for Predictive Text

This module implements several advanced techniques to boost prediction accuracy:
1. Context Augmentation (like SMOTE for text)
2. Dynamic Smoothing
3. Semantic Context Matching
4. Adaptive Ensemble Weighting
5. Advanced Feature Engineering
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import re
from itertools import permutations, combinations
import random


class ContextAugmentor:
    """
    Augments training data similar to how SMOTE augments imbalanced datasets.
    Creates synthetic contexts to improve model robustness.
    """
    
    def __init__(self):
        self.synonym_dict = self._build_synonym_dict()
        self.pattern_cache = {}
        
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """Build a basic synonym dictionary for common words."""
        return {
            # Common words with synonyms
            'said': ['stated', 'mentioned', 'declared', 'expressed', 'remarked'],
            'good': ['excellent', 'great', 'fine', 'wonderful', 'superb'],
            'bad': ['poor', 'terrible', 'awful', 'horrible', 'dreadful'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'petite'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'slow': ['sluggish', 'gradual', 'delayed', 'leisurely', 'unhurried'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'content'],
            'sad': ['sorrowful', 'melancholy', 'dejected', 'gloomy', 'depressed'],
            'important': ['significant', 'crucial', 'vital', 'essential', 'critical'],
            'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive', 'magnificent'],
            'interesting': ['fascinating', 'captivating', 'engaging', 'intriguing', 'compelling'],
            'difficult': ['challenging', 'hard', 'tough', 'complex', 'demanding'],
            'easy': ['simple', 'effortless', 'straightforward', 'uncomplicated', 'basic'],
            'new': ['recent', 'fresh', 'novel', 'modern', 'contemporary'],
            'old': ['ancient', 'aged', 'vintage', 'mature', 'elderly'],
            'many': ['numerous', 'several', 'multiple', 'various', 'countless'],
            'few': ['some', 'several', 'limited', 'scarce', 'minimal']
        }
    
    def augment_contexts(self, contexts: List[Tuple[str, ...]], target_words: List[str], 
                        augmentation_factor: float = 2.0) -> Tuple[List[Tuple[str, ...]], List[str]]:
        """
        Augment training contexts to improve model robustness.
        
        Args:
            contexts: Original context tuples
            target_words: Corresponding target words
            augmentation_factor: How many times to multiply the data
            
        Returns:
            Tuple of (augmented_contexts, augmented_targets)
        """
        print(f"Augmenting {len(contexts)} contexts with factor {augmentation_factor}...")
        
        augmented_contexts = list(contexts)
        augmented_targets = list(target_words)
        
        target_size = int(len(contexts) * augmentation_factor)
        
        while len(augmented_contexts) < target_size:
            # Random selection for augmentation
            idx = random.randint(0, len(contexts) - 1)
            original_context = contexts[idx]
            original_target = target_words[idx]
            
            # Apply different augmentation techniques
            augmented = self._apply_augmentation_techniques(original_context)
            
            for aug_context in augmented:
                if len(augmented_contexts) >= target_size:
                    break
                augmented_contexts.append(aug_context)
                augmented_targets.append(original_target)
        
        print(f"Augmented to {len(augmented_contexts)} contexts")
        return augmented_contexts, augmented_targets
    
    def _apply_augmentation_techniques(self, context: Tuple[str, ...]) -> List[Tuple[str, ...]]:
        """Apply various augmentation techniques to a single context."""
        augmented = []
        
        # 1. Synonym replacement
        syn_context = self._synonym_replacement(context)
        if syn_context != context:
            augmented.append(syn_context)
        
        # 2. Word order variation (for flexible contexts)
        if len(context) >= 3:
            order_context = self._word_order_variation(context)
            if order_context != context:
                augmented.append(order_context)
        
        # 3. Context truncation (shorter contexts)
        if len(context) > 2:
            trunc_context = context[1:]  # Remove first word
            augmented.append(trunc_context)
        
        # 4. Context expansion (add common prefixes)
        expanded_context = self._context_expansion(context)
        if expanded_context != context:
            augmented.append(expanded_context)
        
        return augmented
    
    def _synonym_replacement(self, context: Tuple[str, ...]) -> Tuple[str, ...]:
        """Replace words with synonyms."""
        new_context = list(context)
        
        for i, word in enumerate(context):
            if word.lower() in self.synonym_dict and random.random() < 0.3:
                synonyms = self.synonym_dict[word.lower()]
                new_context[i] = random.choice(synonyms)
        
        return tuple(new_context)
    
    def _word_order_variation(self, context: Tuple[str, ...]) -> Tuple[str, ...]:
        """Slight word order variations for flexible contexts."""
        if len(context) < 3:
            return context
        
        # Only swap adjacent words sometimes
        if random.random() < 0.2:
            context_list = list(context)
            swap_idx = random.randint(0, len(context_list) - 2)
            context_list[swap_idx], context_list[swap_idx + 1] = context_list[swap_idx + 1], context_list[swap_idx]
            return tuple(context_list)
        
        return context
    
    def _context_expansion(self, context: Tuple[str, ...]) -> Tuple[str, ...]:
        """Add common prefixes to expand context."""
        common_prefixes = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'our', 'his', 'her']
        
        if random.random() < 0.2 and len(context) < 5:
            prefix = random.choice(common_prefixes)
            return (prefix,) + context
        
        return context


class AdaptiveSmoother:
    """
    Advanced smoothing that adapts based on context frequency and word rarity.
    """
    
    def __init__(self):
        self.word_frequencies = defaultdict(int)
        self.context_frequencies = defaultdict(int)
        self.total_words = 0
        
    def train(self, contexts: List[Tuple[str, ...]], targets: List[str]):
        """Train the adaptive smoother."""
        print("Training adaptive smoother...")
        
        for context, target in zip(contexts, targets):
            self.context_frequencies[context] += 1
            self.word_frequencies[target] += 1
            self.total_words += 1
    
    def get_smoothed_probability(self, context: Tuple[str, ...], word: str, 
                                raw_count: int, context_count: int) -> float:
        """
        Get adaptively smoothed probability based on word rarity and context frequency.
        """
        if context_count == 0:
            return self._get_backoff_probability(context, word)
        
        # Base probability
        base_prob = raw_count / context_count
        
        # Adaptive smoothing factors
        word_rarity = self._get_word_rarity_factor(word)
        context_reliability = self._get_context_reliability_factor(context)
        
        # Apply adaptive smoothing
        alpha = 0.1 * word_rarity * context_reliability
        smoothed_prob = (1 - alpha) * base_prob + alpha * self._get_backoff_probability(context, word)
        
        return smoothed_prob
    
    def _get_word_rarity_factor(self, word: str) -> float:
        """Calculate rarity factor for the word (rarer words get more smoothing)."""
        word_freq = self.word_frequencies[word]
        if word_freq == 0:
            return 1.0
        
        # Normalize frequency and invert (rarer words get higher factor)
        relative_freq = word_freq / self.total_words
        rarity_factor = 1.0 / (1.0 + 100 * relative_freq)
        return min(1.0, max(0.1, rarity_factor))
    
    def _get_context_reliability_factor(self, context: Tuple[str, ...]) -> float:
        """Calculate reliability factor for the context."""
        context_freq = self.context_frequencies[context]
        if context_freq == 0:
            return 1.0
        
        # More frequent contexts are more reliable, need less smoothing
        reliability_factor = 1.0 / (1.0 + np.log(context_freq + 1))
        return min(1.0, max(0.1, reliability_factor))
    
    def _get_backoff_probability(self, context: Tuple[str, ...], word: str) -> float:
        """Get backoff probability for unseen combinations."""
        if len(context) > 1:
            # Try shorter context
            shorter_context = context[1:]
            shorter_count = self.context_frequencies[shorter_context]
            if shorter_count > 0:
                return self.word_frequencies[word] / (shorter_count * self.total_words)
        
        # Fall back to unigram probability
        return self.word_frequencies[word] / self.total_words if self.total_words > 0 else 1e-10


class SemanticContextMatcher:
    """
    Uses semantic similarity to find related contexts for better predictions.
    """
    
    def __init__(self):
        self.context_vectors = {}
        self.word_cooccurrence = defaultdict(lambda: defaultdict(int))
        
    def train(self, contexts: List[Tuple[str, ...]], targets: List[str]):
        """Train semantic context matcher."""
        print("Training semantic context matcher...")
        
        # Build word co-occurrence matrix
        for context, target in zip(contexts, targets):
            for word1 in context:
                for word2 in context:
                    if word1 != word2:
                        self.word_cooccurrence[word1][word2] += 1
                self.word_cooccurrence[word1][target] += 1
    
    def get_semantic_candidates(self, context: Tuple[str, ...], top_k: int = 10) -> List[str]:
        """Get semantically similar words based on context."""
        if not context:
            return []
        
        # Calculate semantic scores for all words
        word_scores = defaultdict(float)
        
        for context_word in context:
            if context_word in self.word_cooccurrence:
                for candidate_word, cooccur_count in self.word_cooccurrence[context_word].items():
                    word_scores[candidate_word] += cooccur_count
        
        # Sort by score and return top candidates
        sorted_candidates = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_candidates[:top_k]]


class IntelligentEnsemble:
    """
    Advanced ensemble that adapts weights based on context and performance.
    """
    
    def __init__(self, models: List, base_weights: Optional[List[float]] = None):
        self.models = models
        self.base_weights = base_weights or [1.0 / len(models)] * len(models)
        self.adaptive_weights = defaultdict(lambda: list(self.base_weights))
        self.performance_history = defaultdict(list)
        
    def predict_with_adaptive_weighting(self, context: Tuple[str, ...], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Make predictions with context-adaptive ensemble weighting.
        """
        # Get predictions from all models
        all_predictions = []
        model_predictions = []
        
        for i, model in enumerate(self.models):
            try:
                predictions = model.predict(context, k=top_k)
                model_predictions.append(predictions)
            except:
                model_predictions.append([])
        
        # Get adaptive weights for this context type
        context_key = self._get_context_key(context)
        weights = self.adaptive_weights[context_key]
        
        # Combine predictions with adaptive weights
        word_scores = defaultdict(float)
        
        for i, (predictions, weight) in enumerate(zip(model_predictions, weights)):
            for word, prob in predictions:
                word_scores[word] += prob * weight
        
        # Normalize and return top predictions
        if word_scores:
            total_score = sum(word_scores.values())
            if total_score > 0:
                normalized_scores = [(word, score/total_score) for word, score in word_scores.items()]
                return sorted(normalized_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        return []
    
    def update_performance(self, context: Tuple[str, ...], predicted_word: str, actual_word: str):
        """Update performance history to adapt weights."""
        context_key = self._get_context_key(context)
        
        # Check which models predicted correctly
        for i, model in enumerate(self.models):
            try:
                predictions = model.predict(context, k=5)
                predicted_words = [word for word, prob in predictions]
                
                # Award points for correct predictions
                if actual_word in predicted_words:
                    position = predicted_words.index(actual_word) + 1
                    score = 1.0 / position  # Higher score for better position
                else:
                    score = 0.0
                
                self.performance_history[context_key].append((i, score))
            except:
                continue
        
        # Update weights based on recent performance
        self._update_adaptive_weights(context_key)
    
    def _get_context_key(self, context: Tuple[str, ...]) -> str:
        """Get a key for context type to maintain separate weights."""
        # Classify context by length and content type
        length_key = f"len_{len(context)}"
        
        # Simple content classification
        if any(word.istitle() for word in context):
            content_key = "proper_nouns"
        elif any(word.isdigit() for word in context):
            content_key = "numeric"
        elif any(len(word) > 8 for word in context):
            content_key = "technical"
        else:
            content_key = "general"
        
        return f"{length_key}_{content_key}"
    
    def _update_adaptive_weights(self, context_key: str):
        """Update weights based on recent performance."""
        if len(self.performance_history[context_key]) < 10:
            return
        
        # Use recent performance (last 50 evaluations)
        recent_performance = self.performance_history[context_key][-50:]
        
        # Calculate average performance per model
        model_scores = defaultdict(list)
        for model_idx, score in recent_performance:
            model_scores[model_idx].append(score)
        
        # Update weights based on performance
        new_weights = []
        total_performance = 0
        
        for i in range(len(self.models)):
            if i in model_scores:
                avg_score = np.mean(model_scores[i])
                total_performance += avg_score
                new_weights.append(avg_score)
            else:
                new_weights.append(0.1)  # Minimum weight
        
        # Normalize weights
        if total_performance > 0:
            new_weights = [w / total_performance for w in new_weights]
        else:
            new_weights = self.base_weights
        
        # Smooth transition (mix with previous weights)
        alpha = 0.3
        current_weights = self.adaptive_weights[context_key]
        self.adaptive_weights[context_key] = [
            alpha * new_w + (1 - alpha) * curr_w 
            for new_w, curr_w in zip(new_weights, current_weights)
        ]


class AccuracyBooster:
    """
    Main class that combines all accuracy improvement techniques.
    """
    
    def __init__(self):
        self.augmentor = ContextAugmentor()
        self.smoother = AdaptiveSmoother()
        self.semantic_matcher = SemanticContextMatcher()
        self.intelligent_ensemble = None
        
    def enhance_training_data(self, contexts: List[Tuple[str, ...]], targets: List[str], 
                            augmentation_factor: float = 2.0) -> Tuple[List[Tuple[str, ...]], List[str]]:
        """Enhance training data with augmentation techniques."""
        print("Enhancing training data for better accuracy...")
        
        # Apply context augmentation
        aug_contexts, aug_targets = self.augmentor.augment_contexts(
            contexts, targets, augmentation_factor
        )
        
        # Train adaptive smoother
        self.smoother.train(aug_contexts, aug_targets)
        
        # Train semantic matcher
        self.semantic_matcher.train(aug_contexts, aug_targets)
        
        print(f"Enhanced from {len(contexts)} to {len(aug_contexts)} training examples")
        return aug_contexts, aug_targets
    
    def create_intelligent_ensemble(self, models: List) -> IntelligentEnsemble:
        """Create an intelligent ensemble with adaptive weighting."""
        self.intelligent_ensemble = IntelligentEnsemble(models)
        return self.intelligent_ensemble
    
    def get_enhanced_predictions(self, models: List, context: Tuple[str, ...], 
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """Get enhanced predictions using all improvement techniques."""
        if self.intelligent_ensemble is None:
            self.intelligent_ensemble = IntelligentEnsemble(models)
        
        # Get adaptive ensemble predictions
        ensemble_predictions = self.intelligent_ensemble.predict_with_adaptive_weighting(context, top_k)
        
        # Get semantic candidates
        semantic_candidates = self.semantic_matcher.get_semantic_candidates(context, top_k)
        
        # Combine predictions with semantic boost
        enhanced_predictions = {}
        
        # Add ensemble predictions
        for word, prob in ensemble_predictions:
            enhanced_predictions[word] = prob
        
        # Boost semantic candidates
        for word in semantic_candidates:
            if word in enhanced_predictions:
                enhanced_predictions[word] *= 1.2  # 20% boost for semantic similarity
            else:
                enhanced_predictions[word] = 0.1  # Add with small probability
        
        # Sort and return top predictions
        sorted_predictions = sorted(enhanced_predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_k]