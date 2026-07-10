"""
N-gram Model Implementation

This module implements n-gram based Markov models for text prediction.
Supports bigram, trigram, and higher-order n-gram models with efficient
storage and probability calculation.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Union
import pickle
import json
import numpy as np
from .preprocessor import TextPreprocessor


class NgramModel:
    """
    N-gram based Markov model for text prediction.
    
    This class implements the core n-gram model functionality including:
    - N-gram frequency counting
    - Conditional probability calculation
    - Model training and persistence
    - Vocabulary management
    """
    
    def __init__(self, n: int = 2, smoothing_alpha: float = 0.01):
        """
        Initialize the N-gram model.
        
        Args:
            n: Order of the n-gram model (2 for bigram, 3 for trigram, etc.)
            smoothing_alpha: Alpha parameter for Laplace smoothing
        """
        if n < 1:
            raise ValueError("N-gram order must be at least 1")
            
        self.n = n
        self.smoothing_alpha = smoothing_alpha
        
        # N-gram frequency counters
        self.ngram_counts = defaultdict(int)  # N-gram -> count
        self.context_counts = defaultdict(int)  # (n-1)-gram context -> count
        
        # Vocabulary
        self.vocabulary = set()
        
        # Model statistics
        self.total_ngrams = 0
        self.vocab_size = 0
        
        # Training status
        self.is_trained = False
        
        # Preprocessor
        self.preprocessor = TextPreprocessor(
            remove_punctuation=True,
            remove_stopwords=False,
            handle_numbers=True,
            preserve_sentence_boundaries=True
        )
    
    def _extract_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from a list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of n-gram tuples
        """
        if len(tokens) < self.n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _extract_contexts(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """
        Extract (n-1)-gram contexts from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of context tuples
        """
        if self.n == 1:
            return [()]  # Unigram has empty context
        
        if len(tokens) < self.n - 1:
            return []
        
        contexts = []
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            contexts.append(context)
        
        return contexts
    
    def train_from_tokens(self, tokens: List[str]):
        """
        Train the model from a list of preprocessed tokens.
        
        Args:
            tokens: List of preprocessed tokens
        """
        if not tokens:
            raise ValueError("Token list cannot be empty")
        
        # Update vocabulary
        self.vocabulary.update(tokens)
        self.vocab_size = len(self.vocabulary)
        
        # Extract n-grams and contexts
        ngrams = self._extract_ngrams(tokens)
        contexts = self._extract_contexts(tokens)
        
        # Count n-grams
        for ngram in ngrams:
            self.ngram_counts[ngram] += 1
            self.total_ngrams += 1
        
        # Count contexts
        for context in contexts:
            self.context_counts[context] += 1
        
        self.is_trained = True
    
    def train_from_text(self, text: str):
        """
        Train the model from raw text.
        
        Args:
            text: Raw text string
        """
        tokens = self.preprocessor.preprocess_text(text)
        self.train_from_tokens(tokens)
    
    def train_from_file(self, file_path: str, encoding: str = 'utf-8'):
        """
        Train the model from a text file.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding
        """
        tokens = self.preprocessor.preprocess_file(file_path, encoding)
        self.train_from_tokens(tokens)
    
    def train_from_corpus(self, texts: List[str]):
        """
        Train the model from a corpus of texts.
        
        Args:
            texts: List of text documents
        """
        all_tokens = self.preprocessor.preprocess_corpus(texts)
        self.train_from_tokens(all_tokens)
    
    def get_ngram_probability(self, ngram: Tuple[str, ...], smoothed: bool = True) -> float:
        """
        Calculate the probability of an n-gram.
        
        Args:
            ngram: N-gram tuple
            smoothed: Whether to apply Laplace smoothing
            
        Returns:
            Probability of the n-gram
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating probabilities")
        
        if len(ngram) != self.n:
            raise ValueError(f"N-gram length must be {self.n}")
        
        if self.n == 1:
            # Unigram probability
            if smoothed:
                return (self.ngram_counts[ngram] + self.smoothing_alpha) / \
                       (self.total_ngrams + self.smoothing_alpha * self.vocab_size)
            else:
                return self.ngram_counts[ngram] / self.total_ngrams if self.total_ngrams > 0 else 0
        else:
            # Higher-order n-gram probability
            context = ngram[:-1]
            context_count = self.context_counts[context]
            
            if context_count == 0:
                return 0.0
            
            if smoothed:
                return (self.ngram_counts[ngram] + self.smoothing_alpha) / \
                       (context_count + self.smoothing_alpha * self.vocab_size)
            else:
                return self.ngram_counts[ngram] / context_count
    
    def get_conditional_probability(self, word: str, context: Tuple[str, ...], smoothed: bool = True) -> float:
        """
        Calculate P(word | context).
        
        Args:
            word: Target word
            context: Context tuple (n-1 words)
            smoothed: Whether to apply smoothing
            
        Returns:
            Conditional probability
        """
        if len(context) != self.n - 1:
            raise ValueError(f"Context length must be {self.n - 1}")
        
        ngram = context + (word,)
        return self.get_ngram_probability(ngram, smoothed)
    
    def get_possible_continuations(self, context: Tuple[str, ...]) -> List[Tuple[str, float]]:
        """
        Get all possible word continuations for a given context with their probabilities.
        
        Args:
            context: Context tuple
            
        Returns:
            List of (word, probability) pairs sorted by probability (descending)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting continuations")
        
        # Handle flexible context lengths
        required_length = self.n - 1
        if len(context) > required_length:
            context = context[-required_length:]
        elif len(context) < required_length:
            if len(context) == 0:
                # For empty context, return most frequent unigrams
                return self._get_most_frequent_words(10)
        
        continuations = []
        
        # Find all n-grams that start with the given context
        for ngram in self.ngram_counts:
            if len(ngram) == self.n and ngram[:-1] == context:
                word = ngram[-1]
                prob = self.get_conditional_probability(word, context)
                continuations.append((word, prob))
        
        # Sort by probability (descending)
        continuations.sort(key=lambda x: x[1], reverse=True)
        
        return continuations
    
    def get_top_predictions(self, context: Tuple[str, ...], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k word predictions for a given context.
        
        Args:
            context: Context tuple
            top_k: Number of top predictions to return
            
        Returns:
            List of top-k (word, probability) pairs
        """
        continuations = self.get_possible_continuations(context)
        return continuations[:top_k]
    
    def predict(self, context: Union[List[str], Tuple[str, ...]], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next words for the given context (alias for get_top_predictions).
        
        Args:
            context: Context as list or tuple of strings
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples sorted by probability
        """
        # Convert context to tuple if it's a list
        if isinstance(context, list):
            context = tuple(context)
        
        # Adjust context length to match model requirements
        required_length = self.n - 1
        
        if len(context) > required_length:
            # Take the last (n-1) words
            context = context[-required_length:]
        elif len(context) < required_length:
            # Pad with special tokens or handle gracefully
            if len(context) == 0:
                # For empty context, return most frequent words
                return self._get_most_frequent_words(top_k)
            # Use shorter context (fallback to available context)
            # This will work for n-grams where we have data
        
        try:
            return self.get_top_predictions(context, top_k)
        except ValueError:
            # If context still doesn't work, fallback to most frequent words
            return self._get_most_frequent_words(top_k)
    
    def _get_most_frequent_words(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get most frequent words in vocabulary as fallback.
        
        Args:
            top_k: Number of words to return
            
        Returns:
            List of (word, probability) tuples
        """
        if not self.vocabulary:
            return []
        
        # Count word frequencies from unigrams
        word_counts = defaultdict(int)
        for ngram in self.ngram_counts:
            if len(ngram) == 1:  # unigram
                word_counts[ngram[0]] += self.ngram_counts[ngram]
        
        if not word_counts:
            # If no unigrams, extract from n-grams
            for ngram in self.ngram_counts:
                for word in ngram:
                    word_counts[word] += self.ngram_counts[ngram]
        
        # Convert to probabilities
        total = sum(word_counts.values())
        if total == 0:
            return []
        
        word_probs = [(word, count/total) for word, count in word_counts.items()]
        word_probs.sort(key=lambda x: x[1], reverse=True)
        
        return word_probs[:top_k]
    
    def generate_text(self, start_context: Optional[Tuple[str, ...]] = None, 
                     max_length: int = 50, temperature: float = 1.0) -> List[str]:
        """
        Generate text using the trained model.
        
        Args:
            start_context: Starting context (if None, uses sentence start)
            max_length: Maximum length of generated text
            temperature: Sampling temperature (1.0 = normal, >1.0 = more random)
            
        Returns:
            Generated text as list of words
        """
        import random
        import numpy as np
        
        if not self.is_trained:
            raise ValueError("Model must be trained before generating text")
        
        # Initialize context
        if start_context is None:
            if self.n > 1:
                context = tuple([self.preprocessor.sentence_start_token] * (self.n - 1))
            else:
                context = ()
        else:
            if len(start_context) != self.n - 1:
                raise ValueError(f"Start context length must be {self.n - 1}")
            context = start_context
        
        generated = list(context) if context else []
        
        for _ in range(max_length):
            continuations = self.get_possible_continuations(context)
            
            if not continuations:
                break
            
            # Apply temperature sampling
            if temperature == 0:
                # Greedy selection
                next_word = continuations[0][0]
            else:
                # Temperature-scaled sampling
                words, probs = zip(*continuations)
                probs = np.array(probs)
                
                if temperature != 1.0:
                    probs = probs ** (1.0 / temperature)
                    probs = probs / np.sum(probs)
                
                next_word = np.random.choice(words, p=probs)
            
            generated.append(next_word)
            
            # Update context for next prediction
            if self.n > 1:
                context = context[1:] + (next_word,)
            
            # Stop at sentence end
            if next_word == self.preprocessor.sentence_end_token:
                break
        
        return generated
    
    def save_model(self, file_path: str):
        """
        Save the trained model to a file.
        
        Args:
            file_path: Path to save the model
        """
        model_data = {
            'n': self.n,
            'smoothing_alpha': self.smoothing_alpha,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'vocabulary': list(self.vocabulary),
            'total_ngrams': self.total_ngrams,
            'vocab_size': self.vocab_size,
            'is_trained': self.is_trained
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, file_path: str):
        """
        Load a trained model from a file.
        
        Args:
            file_path: Path to the saved model
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n = model_data['n']
        self.smoothing_alpha = model_data['smoothing_alpha']
        self.ngram_counts = defaultdict(int, model_data['ngram_counts'])
        self.context_counts = defaultdict(int, model_data['context_counts'])
        self.vocabulary = set(model_data['vocabulary'])
        self.total_ngrams = model_data['total_ngrams']
        self.vocab_size = model_data['vocab_size']
        self.is_trained = model_data['is_trained']
    
    def get_model_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get model statistics.
        
        Returns:
            Dictionary with model statistics
        """
        return {
            'n_gram_order': self.n,
            'vocabulary_size': self.vocab_size,
            'total_ngrams': self.total_ngrams,
            'unique_ngrams': len(self.ngram_counts),
            'unique_contexts': len(self.context_counts),
            'smoothing_alpha': self.smoothing_alpha,
            'is_trained': self.is_trained,
            'perplexity_on_training': self.calculate_perplexity() if self.is_trained else None
        }
    
    def calculate_perplexity(self, test_tokens: Optional[List[str]] = None) -> float:
        """
        Calculate perplexity of the model on test data.
        
        Args:
            test_tokens: Test tokens (if None, uses training data)
            
        Returns:
            Perplexity value
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating perplexity")
        
        if test_tokens is None:
            # Use training data for perplexity calculation
            test_ngrams = list(self.ngram_counts.keys())
        else:
            test_ngrams = self._extract_ngrams(test_tokens)
        
        if not test_ngrams:
            return float('inf')
        
        log_prob_sum = 0
        for ngram in test_ngrams:
            prob = self.get_ngram_probability(ngram, smoothed=True)
            if prob > 0:
                log_prob_sum += -np.log2(prob)
            else:
                return float('inf')
        
        perplexity = 2 ** (log_prob_sum / len(test_ngrams))
        return perplexity
