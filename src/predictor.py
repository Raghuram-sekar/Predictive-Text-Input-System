"""
Prediction Logic Module

This module implements the prediction logic for the Predictive Text Input System.
It provides high-level interfaces for making text predictions using trained n-gram models.
"""

from typing import List, Tuple, Optional, Dict
import re
import numpy as np
from .ngram_model import NgramModel


class Predictor:
    """
    High-level predictor class for text prediction using n-gram models.
    
    This class provides user-friendly interfaces for making predictions,
    handling various input formats, and managing prediction contexts.
    """
    
    def __init__(self, model: NgramModel):
        """
        Initialize the predictor with a trained n-gram model.
        
        Args:
            model: Trained NgramModel instance
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before use")
        
        self.model = model
        self.preprocessor = model.preprocessor
    
    def predict(self, 
                text_input: str, 
                top_k: int = 5, 
                include_probabilities: bool = False) -> List[str]:
        """
        Predict the next word(s) given input text.
        
        Args:
            text_input: Input text string
            top_k: Number of predictions to return
            include_probabilities: Whether to return probabilities with predictions
            
        Returns:
            List of predicted words or (word, probability) tuples
        """
        # Preprocess the input text
        tokens = self.preprocessor.preprocess_text(text_input)
        
        if not tokens:
            # No valid tokens, return empty predictions
            return []
        
        # Extract context for prediction
        context = self._get_prediction_context(tokens)
        
        # Get predictions from model
        predictions = self.model.get_top_predictions(context, top_k)
        
        if include_probabilities:
            return predictions
        else:
            return [word for word, _ in predictions]
    
    def predict_from_context(self, 
                           context_words: List[str], 
                           top_k: int = 5, 
                           include_probabilities: bool = False) -> List[str]:
        """
        Predict the next word given a specific context.
        
        Args:
            context_words: List of context words
            top_k: Number of predictions to return
            include_probabilities: Whether to return probabilities
            
        Returns:
            List of predicted words or (word, probability) tuples
        """
        # Validate context length
        required_context_length = self.model.n - 1
        
        if len(context_words) < required_context_length:
            # Pad context if too short
            padding = [self.preprocessor.sentence_start_token] * (required_context_length - len(context_words))
            context_words = padding + context_words
        elif len(context_words) > required_context_length:
            # Truncate context if too long
            context_words = context_words[-required_context_length:]
        
        context = tuple(context_words)
        predictions = self.model.get_top_predictions(context, top_k)
        
        if include_probabilities:
            return predictions
        else:
            return [word for word, _ in predictions]
    
    def _get_prediction_context(self, tokens: List[str]) -> Tuple[str, ...]:
        """
        Extract the appropriate context for prediction from tokens.
        
        Args:
            tokens: List of preprocessed tokens
            
        Returns:
            Context tuple for prediction
        """
        required_length = self.model.n - 1
        
        if required_length == 0:
            # Unigram model has no context
            return ()
        
        if len(tokens) >= required_length:
            # Use the last n-1 tokens as context
            return tuple(tokens[-required_length:])
        else:
            # Pad with sentence start tokens if needed
            padding = [self.preprocessor.sentence_start_token] * (required_length - len(tokens))
            return tuple(padding + tokens)
    
    def predict_sentence_completion(self, 
                                  partial_sentence: str, 
                                  max_words: int = 10, 
                                  temperature: float = 1.0) -> str:
        """
        Complete a partial sentence using the model.
        
        Args:
            partial_sentence: Incomplete sentence
            max_words: Maximum words to generate
            temperature: Sampling temperature
            
        Returns:
            Completed sentence
        """
        # Preprocess input
        tokens = self.preprocessor.preprocess_text(partial_sentence)
        
        # Get completion using model generation
        context = self._get_prediction_context(tokens)
        
        # Generate continuation
        generated = self.model.generate_text(
            start_context=context,
            max_length=max_words,
            temperature=temperature
        )
        
        # Remove the context part and special tokens
        if context:
            generated = generated[len(context):]
        
        # Filter out special tokens
        special_tokens = {
            self.preprocessor.sentence_start_token,
            self.preprocessor.sentence_end_token
        }
        
        completion_words = [word for word in generated if word not in special_tokens]
        
        return ' '.join(completion_words)
    
    def get_word_suggestions(self, 
                           partial_word: str, 
                           context: str = "", 
                           top_k: int = 5) -> List[str]:
        """
        Get word suggestions that start with the given partial word.
        
        Args:
            partial_word: Partial word to complete
            context: Context text before the partial word
            top_k: Number of suggestions to return
            
        Returns:
            List of word suggestions
        """
        # Get all possible continuations for the context
        if context:
            context_tokens = self.preprocessor.preprocess_text(context)
            prediction_context = self._get_prediction_context(context_tokens)
        else:
            prediction_context = tuple([self.preprocessor.sentence_start_token] * (self.model.n - 1))
        
        # Get all possible continuations
        continuations = self.model.get_possible_continuations(prediction_context)
        
        # Filter words that start with the partial word
        partial_lower = partial_word.lower()
        matching_words = [
            (word, prob) for word, prob in continuations 
            if word.lower().startswith(partial_lower)
        ]
        
        # Sort by probability and return top-k
        matching_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in matching_words[:top_k]]
    
    def evaluate_text_probability(self, text: str) -> float:
        """
        Calculate the probability of a given text according to the model.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Log probability of the text
        """
        tokens = self.preprocessor.preprocess_text(text)
        ngrams = self.model._extract_ngrams(tokens)
        
        if not ngrams:
            return float('-inf')
        
        log_prob = 0.0
        for ngram in ngrams:
            prob = self.model.get_ngram_probability(ngram, smoothed=True)
            if prob > 0:
                log_prob += np.log(prob)
            else:
                return float('-inf')
        
        return log_prob
    
    def compare_texts(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Compare multiple texts and rank them by probability.
        
        Args:
            texts: List of texts to compare
            
        Returns:
            List of (text, probability) tuples sorted by probability
        """
        text_probs = []
        for text in texts:
            prob = self.evaluate_text_probability(text)
            text_probs.append((text, prob))
        
        # Sort by probability (descending)
        text_probs.sort(key=lambda x: x[1], reverse=True)
        return text_probs


class InteractivePredictionSession:
    """
    Interactive session for continuous text prediction.
    
    This class maintains state across multiple prediction requests,
    allowing for context-aware predictions in interactive scenarios.
    """
    
    def __init__(self, predictor: Predictor):
        """
        Initialize an interactive prediction session.
        
        Args:
            predictor: Predictor instance
        """
        self.predictor = predictor
        self.session_history = []
        self.current_context = []
        self.max_context_length = predictor.model.n - 1
    
    def add_text(self, text: str):
        """
        Add text to the session context.
        
        Args:
            text: Text to add to context
        """
        tokens = self.predictor.preprocessor.preprocess_text(text)
        self.session_history.extend(tokens)
        
        # Update current context
        if self.max_context_length > 0:
            all_tokens = self.session_history
            if len(all_tokens) >= self.max_context_length:
                self.current_context = all_tokens[-self.max_context_length:]
            else:
                # Pad with start tokens if needed
                padding_needed = self.max_context_length - len(all_tokens)
                padding = [self.predictor.preprocessor.sentence_start_token] * padding_needed
                self.current_context = padding + all_tokens
        else:
            self.current_context = []
    
    def predict_next(self, top_k: int = 5) -> List[str]:
        """
        Predict the next word based on current session context.
        
        Args:
            top_k: Number of predictions to return
            
        Returns:
            List of predicted words
        """
        if self.max_context_length == 0:
            # Unigram model
            context = ()
        else:
            context = tuple(self.current_context)
        
        predictions = self.predictor.model.get_top_predictions(context, top_k)
        return [word for word, _ in predictions]
    
    def accept_prediction(self, word: str):
        """
        Accept a prediction and update the session context.
        
        Args:
            word: Accepted word
        """
        self.session_history.append(word)
        
        # Update current context
        if self.max_context_length > 0:
            self.current_context.append(word)
            if len(self.current_context) > self.max_context_length:
                self.current_context = self.current_context[-self.max_context_length:]
    
    def reset_session(self):
        """Reset the session history and context."""
        self.session_history = []
        self.current_context = []
    
    def get_session_text(self) -> str:
        """
        Get the current session text.
        
        Returns:
            Session text as string
        """
        # Filter out special tokens for display
        special_tokens = {
            self.predictor.preprocessor.sentence_start_token,
            self.predictor.preprocessor.sentence_end_token
        }
        
        display_tokens = [
            token for token in self.session_history 
            if token not in special_tokens
        ]
        
        return ' '.join(display_tokens)
