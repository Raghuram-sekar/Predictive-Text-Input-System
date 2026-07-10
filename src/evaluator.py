"""
Evaluation Framework Module

This module provides comprehensive evaluation tools for the Predictive Text Input System,
including accuracy metrics, performance benchmarks, and robustness testing.
"""

import time
import random
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
from .ngram_model import NgramModel
from .predictor import Predictor
from .preprocessor import TextPreprocessor


class ModelEvaluator:
    """
    Comprehensive evaluation framework for n-gram models and predictors.
    """
    
    def __init__(self, model: NgramModel, predictor: Optional[Predictor] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained NgramModel instance
            predictor: Optional Predictor instance
        """
        self.model = model
        self.predictor = predictor or Predictor(model)
        self.preprocessor = model.preprocessor
    
    def evaluate_prediction_accuracy(self, 
                                   test_data: List[str], 
                                   top_k: int = 5) -> Dict[str, float]:
        """
        Evaluate prediction accuracy on test data.
        
        Args:
            test_data: List of test sentences
            top_k: Consider top-k predictions for accuracy
            
        Returns:
            Dictionary with accuracy metrics
        """
        total_predictions = 0
        correct_predictions = [0] * top_k  # For top-1, top-2, ..., top-k accuracy
        
        for sentence in test_data:
            tokens = self.preprocessor.preprocess_text(sentence)
            
            # Skip sentences that are too short
            if len(tokens) < self.model.n:
                continue
            
            # Evaluate each position in the sentence
            for i in range(self.model.n - 1, len(tokens)):
                # Get context
                context = tokens[i - self.model.n + 1:i]
                actual_word = tokens[i]
                
                # Get predictions
                predictions = self.predictor.predict_from_context(context, top_k=top_k)
                
                # Check if actual word is in top-k predictions
                for k in range(min(top_k, len(predictions))):
                    if actual_word in predictions[:k+1]:
                        correct_predictions[k] += 1
                
                total_predictions += 1
        
        # Calculate accuracies
        accuracies = {}
        for k in range(top_k):
            if total_predictions > 0:
                accuracies[f'top_{k+1}_accuracy'] = correct_predictions[k] / total_predictions
            else:
                accuracies[f'top_{k+1}_accuracy'] = 0.0
        
        accuracies['total_predictions'] = total_predictions
        return accuracies
    
    def evaluate_perplexity(self, test_data: List[str]) -> float:
        """
        Calculate perplexity on test data.
        
        Args:
            test_data: List of test sentences
            
        Returns:
            Perplexity value
        """
        total_log_prob = 0.0
        total_ngrams = 0
        
        for sentence in test_data:
            tokens = self.preprocessor.preprocess_text(sentence)
            ngrams = self.model._extract_ngrams(tokens)
            
            for ngram in ngrams:
                prob = self.model.get_ngram_probability(ngram, smoothed=True)
                if prob > 0:
                    total_log_prob += np.log2(prob)
                else:
                    return float('inf')  # If any probability is 0, perplexity is infinite
                total_ngrams += 1
        
        if total_ngrams == 0:
            return float('inf')
        
        return 2 ** (-total_log_prob / total_ngrams)
    
    def evaluate_response_time(self, 
                             test_contexts: List[str], 
                             num_predictions: int = 5, 
                             num_runs: int = 100) -> Dict[str, float]:
        """
        Evaluate prediction response time.
        
        Args:
            test_contexts: List of context strings
            num_predictions: Number of predictions to generate
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with timing statistics
        """
        response_times = []
        
        for _ in range(num_runs):
            context = random.choice(test_contexts)
            
            start_time = time.time()
            predictions = self.predictor.predict(context, top_k=num_predictions)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
        
        return {
            'mean_response_time': np.mean(response_times),
            'median_response_time': np.median(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'std_response_time': np.std(response_times),
            'total_runs': num_runs
        }
    
    def evaluate_robustness(self, test_data: List[str]) -> Dict[str, Any]:
        """
        Evaluate model robustness to noisy inputs.
        
        Args:
            test_data: List of clean test sentences
            
        Returns:
            Dictionary with robustness metrics
        """
        results = {
            'clean_accuracy': None,
            'noisy_accuracy': {},
            'degradation': {}
        }
        
        # Evaluate on clean data
        clean_accuracy = self.evaluate_prediction_accuracy(test_data, top_k=1)
        results['clean_accuracy'] = clean_accuracy['top_1_accuracy']
        
        # Test different noise types
        noise_types = ['typos', 'case_changes', 'extra_spaces', 'punctuation']
        
        for noise_type in noise_types:
            noisy_data = self._add_noise(test_data, noise_type)
            noisy_accuracy = self.evaluate_prediction_accuracy(noisy_data, top_k=1)
            
            results['noisy_accuracy'][noise_type] = noisy_accuracy['top_1_accuracy']
            results['degradation'][noise_type] = results['clean_accuracy'] - noisy_accuracy['top_1_accuracy']
        
        return results
    
    def _add_noise(self, texts: List[str], noise_type: str, noise_level: float = 0.1) -> List[str]:
        """
        Add noise to text data for robustness testing.
        
        Args:
            texts: List of clean texts
            noise_type: Type of noise to add
            noise_level: Probability of applying noise to each character/word
            
        Returns:
            List of noisy texts
        """
        noisy_texts = []
        
        for text in texts:
            if noise_type == 'typos':
                noisy_text = self._add_typos(text, noise_level)
            elif noise_type == 'case_changes':
                noisy_text = self._add_case_changes(text, noise_level)
            elif noise_type == 'extra_spaces':
                noisy_text = self._add_extra_spaces(text, noise_level)
            elif noise_type == 'punctuation':
                noisy_text = self._add_punctuation_noise(text, noise_level)
            else:
                noisy_text = text
            
            noisy_texts.append(noisy_text)
        
        return noisy_texts
    
    def _add_typos(self, text: str, noise_level: float) -> str:
        """Add random character substitutions."""
        chars = list(text)
        for i, char in enumerate(chars):
            if char.isalpha() and random.random() < noise_level:
                # Replace with random letter
                chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return ''.join(chars)
    
    def _add_case_changes(self, text: str, noise_level: float) -> str:
        """Add random case changes."""
        chars = list(text)
        for i, char in enumerate(chars):
            if char.isalpha() and random.random() < noise_level:
                chars[i] = char.upper() if char.islower() else char.lower()
        return ''.join(chars)
    
    def _add_extra_spaces(self, text: str, noise_level: float) -> str:
        """Add extra spaces randomly."""
        chars = list(text)
        result = []
        for char in chars:
            result.append(char)
            if random.random() < noise_level:
                result.append(' ')
        return ''.join(result)
    
    def _add_punctuation_noise(self, text: str, noise_level: float) -> str:
        """Add or remove punctuation randomly."""
        import string
        chars = list(text)
        for i, char in enumerate(chars):
            if random.random() < noise_level:
                if char in string.punctuation:
                    # Remove punctuation
                    chars[i] = ' '
                elif char == ' ':
                    # Add punctuation
                    chars[i] = random.choice('.,!?;:')
        return ''.join(chars)
    
    def evaluate_vocabulary_coverage(self, test_data: List[str]) -> Dict[str, float]:
        """
        Evaluate vocabulary coverage on test data.
        
        Args:
            test_data: List of test sentences
            
        Returns:
            Dictionary with vocabulary coverage metrics
        """
        # Get test vocabulary
        test_tokens = []
        for sentence in test_data:
            tokens = self.preprocessor.preprocess_text(sentence)
            test_tokens.extend(tokens)
        
        test_vocab = set(test_tokens)
        
        # Calculate coverage
        covered_words = test_vocab.intersection(self.model.vocabulary)
        oov_words = test_vocab - self.model.vocabulary
        
        return {
            'test_vocab_size': len(test_vocab),
            'training_vocab_size': len(self.model.vocabulary),
            'covered_words': len(covered_words),
            'oov_words': len(oov_words),
            'coverage_ratio': len(covered_words) / len(test_vocab) if test_vocab else 0,
            'oov_ratio': len(oov_words) / len(test_vocab) if test_vocab else 0
        }
    
    def comprehensive_evaluation(self, 
                               test_data: List[str], 
                               validation_data: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the model.
        
        Args:
            test_data: Test dataset
            validation_data: Optional validation dataset
            
        Returns:
            Comprehensive evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        results = {
            'model_info': self.model.get_model_stats(),
            'accuracy': None,
            'perplexity': None,
            'response_time': None,
            'vocabulary_coverage': None,
            'robustness': None
        }
        
        # Accuracy evaluation
        print("Evaluating prediction accuracy...")
        results['accuracy'] = self.evaluate_prediction_accuracy(test_data, top_k=5)
        
        # Perplexity evaluation
        print("Calculating perplexity...")
        results['perplexity'] = self.evaluate_perplexity(test_data)
        
        # Response time evaluation
        print("Measuring response time...")
        test_contexts = [sentence.split()[:-1] if sentence.split() else [] for sentence in test_data[:50]]
        test_contexts = [' '.join(context) for context in test_contexts if context]
        if test_contexts:
            results['response_time'] = self.evaluate_response_time(test_contexts)
        
        # Vocabulary coverage
        print("Analyzing vocabulary coverage...")
        results['vocabulary_coverage'] = self.evaluate_vocabulary_coverage(test_data)
        
        # Robustness evaluation
        print("Testing robustness...")
        results['robustness'] = self.evaluate_robustness(test_data[:100])  # Use subset for efficiency
        
        print("Evaluation completed!")
        return results
    
    def compare_models(self, other_models: List[NgramModel], test_data: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same test data.
        
        Args:
            other_models: List of other trained models to compare
            test_data: Test dataset
            
        Returns:
            Comparison results for all models
        """
        models = [self.model] + other_models
        model_names = [f"{model.n}-gram" for model in models]
        
        comparison_results = {}
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            print(f"Evaluating {name} model...")
            
            evaluator = ModelEvaluator(model)
            results = {
                'accuracy': evaluator.evaluate_prediction_accuracy(test_data, top_k=1)['top_1_accuracy'],
                'perplexity': evaluator.evaluate_perplexity(test_data),
                'vocab_coverage': evaluator.evaluate_vocabulary_coverage(test_data)['coverage_ratio']
            }
            
            comparison_results[name] = results
        
        return comparison_results
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("=" * 50)
        report.append("PREDICTIVE TEXT SYSTEM EVALUATION REPORT")
        report.append("=" * 50)
        
        # Model Information
        if 'model_info' in results:
            report.append("\nMODEL INFORMATION:")
            for key, value in results['model_info'].items():
                report.append(f"  {key}: {value}")
        
        # Accuracy Results
        if 'accuracy' in results and results['accuracy']:
            report.append("\nPREDICTION ACCURACY:")
            for key, value in results['accuracy'].items():
                if 'accuracy' in key:
                    report.append(f"  {key}: {value:.4f} ({value*100:.2f}%)")
                else:
                    report.append(f"  {key}: {value}")
        
        # Perplexity
        if 'perplexity' in results:
            report.append(f"\nPERPLEXITY: {results['perplexity']:.4f}")
        
        # Response Time
        if 'response_time' in results and results['response_time']:
            report.append("\nRESPONSE TIME:")
            rt = results['response_time']
            report.append(f"  Mean: {rt['mean_response_time']*1000:.2f} ms")
            report.append(f"  Median: {rt['median_response_time']*1000:.2f} ms")
            report.append(f"  Std Dev: {rt['std_response_time']*1000:.2f} ms")
        
        # Vocabulary Coverage
        if 'vocabulary_coverage' in results:
            report.append("\nVOCABULARY COVERAGE:")
            vc = results['vocabulary_coverage']
            report.append(f"  Coverage: {vc['coverage_ratio']:.4f} ({vc['coverage_ratio']*100:.2f}%)")
            report.append(f"  OOV Rate: {vc['oov_ratio']:.4f} ({vc['oov_ratio']*100:.2f}%)")
        
        # Robustness
        if 'robustness' in results and results['robustness']:
            report.append("\nROBUSTNESS ANALYSIS:")
            rob = results['robustness']
            report.append(f"  Clean Accuracy: {rob['clean_accuracy']:.4f}")
            for noise_type, accuracy in rob['noisy_accuracy'].items():
                degradation = rob['degradation'][noise_type]
                report.append(f"  {noise_type}: {accuracy:.4f} (degradation: {degradation:.4f})")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)
