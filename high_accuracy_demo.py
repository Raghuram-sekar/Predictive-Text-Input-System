"""
Modern Smoothing Technique Demo

This demo shows accuracy improvements using modern neural-based smoothing:
1. Transformer-based word embeddings
2. Context-aware semantic smoothing
3. Adaptive interpolation
4. Comparative analysis with traditional methods
"""

import os
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict

# Import components
from data.enhanced_dataset_downloader import EnhancedDatasetDownloader
from src.enhanced_preprocessor import EnhancedPreprocessor
from src.modern_transformer import NeuralInterpolationSmoother
from src.advanced_smoothing import KneserNeySmoother
from src.ngram_model import NgramModel


def prepare_test_data(filepath: str, preprocessor: EnhancedPreprocessor) -> List[Tuple[List[str], str]]:
    """
    Prepare test data for evaluation.
    
    Args:
        filepath: Path to test data file
        preprocessor: Text preprocessor
        
    Returns:
        List of (context, target) pairs
    """
    print("\nPreparing test data...")
    test_pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Preprocess text
    tokens = preprocessor.preprocess_text(text)
    
    # Create test pairs (context, target)
    context_size = 3  # Use trigram context
    for i in range(len(tokens) - context_size):
        context = tokens[i:i+context_size]
        target = tokens[i+context_size]
        test_pairs.append((context, target))
    
    return test_pairs


def evaluate_smoothing_methods(test_data: List[Tuple[List[str], str]], 
                             models: Dict[str, NgramModel]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate and compare different smoothing methods.
    
    Args:
        test_data: List of (context, target) pairs
        models: Dictionary of models with different smoothing
        
    Returns:
        Dictionary of evaluation metrics for each model
    """
    print("\nEvaluating smoothing methods...")
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = defaultdict(float)
        total = len(test_data)
        
        # Track different types of predictions
        correct = 0
        correct_rare = 0  # Correct predictions for rare words
        total_rare = 0   # Total rare words encountered
        mrr = 0.0        # Mean Reciprocal Rank
        
        for context, target in tqdm(test_data):
            predictions = model.predict(context)
            pred_words = [w for w, _ in predictions]
            
            # Check accuracy
            if pred_words and pred_words[0] == target:
                correct += 1
            
            # Calculate MRR
            if target in pred_words:
                rank = pred_words.index(target) + 1
                mrr += 1.0 / rank
            
            # Check rare word handling
            target_freq = model.get_word_frequency(target)
            if target_freq < 5:  # Consider words with frequency < 5 as rare
                total_rare += 1
                if pred_words and pred_words[0] == target:
                    correct_rare += 1
        
        # Calculate metrics
        metrics['accuracy'] = correct / total
        metrics['rare_word_accuracy'] = correct_rare / total_rare if total_rare > 0 else 0
        metrics['mrr'] = mrr / total
        
        results[name] = metrics
        
        # Print detailed results
        print(f"\n{name} Results:")
        print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
        print(f"Rare Word Accuracy: {metrics['rare_word_accuracy']:.2%}")
        print(f"Mean Reciprocal Rank: {metrics['mrr']:.3f}")
    
    return results


def compare_prediction_examples(models: Dict[str, NgramModel], 
                             test_sentences: List[str],
                             preprocessor: EnhancedPreprocessor):
    """
    Compare predictions from different models on example sentences.
    
    Args:
        models: Dictionary of models to compare
        test_sentences: List of test sentences
        preprocessor: Text preprocessor
    """
    print("\nComparing prediction examples:")
    print("=" * 50)
    
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        
        # Preprocess
        tokens = preprocessor.preprocess_text(sentence)
        if len(tokens) < 3:
            continue
            
        # Get context
        context = tokens[-3:]
        print(f"Context: {' '.join(context)}")
        
        # Get predictions from each model
        print("\nPredictions:")
        for name, model in models.items():
            predictions = model.predict(context)
            pred_str = ", ".join([f"{w} ({p:.3f})" for w, p in predictions[:3]])
            print(f"{name:15s}: {pred_str}")
        print("-" * 50)


def main():
    """Main demo function."""
    print("Modern Smoothing Technique Demo")
    print("=" * 50)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize components
    downloader = EnhancedDatasetDownloader()
    preprocessor = EnhancedPreprocessor()
    
    # Download dataset if needed
    print("\nPreparing dataset...")
    corpus_path = downloader.create_enhanced_combined_corpus()
    
    # Read and preprocess corpus
    print("\nPreprocessing text...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = preprocessor.preprocess_text(text)
    
    # Create models with different smoothing
    print("\nTraining models...")
    models = {}
    
    # 1. Traditional Kneser-Ney model
    print("Training Kneser-Ney model...")
    kn_model = NgramModel(n=3)
    kn_model.smoother = KneserNeySmoother()
    kn_model.train_from_tokens(tokens)
    models['kneser_ney'] = kn_model
    
    # 2. Modern Neural model
    print("Training Neural model...")
    neural_model = NgramModel(n=3)
    neural_model.smoother = NeuralInterpolationSmoother()
    neural_model.train_from_tokens(tokens)
    models['neural'] = neural_model
    
    # Prepare test data
    test_data = prepare_test_data("data/test_data.txt", preprocessor)
    
    # Evaluate models
    results = evaluate_smoothing_methods(test_data, models)
    
    # Compare on example sentences
    test_sentences = [
        "The quick brown fox jumps",
        "In machine learning, the model",
        "The neural network learns to",
        "According to the latest research",
        "The implementation of the algorithm"
    ]
    
    compare_prediction_examples(models, test_sentences, preprocessor)
    
    # Print summary
    print("\nSummary of Improvements:")
    print("=" * 50)
    neural_metrics = results['neural']
    kn_metrics = results['kneser_ney']
    
    acc_improvement = (neural_metrics['accuracy'] - kn_metrics['accuracy']) * 100
    rare_improvement = (neural_metrics['rare_word_accuracy'] - kn_metrics['rare_word_accuracy']) * 100
    mrr_improvement = (neural_metrics['mrr'] - kn_metrics['mrr']) * 100
    
    print(f"Overall Accuracy Improvement: {acc_improvement:+.1f}%")
    print(f"Rare Word Accuracy Improvement: {rare_improvement:+.1f}%")
    print(f"MRR Improvement: {mrr_improvement:+.1f}%")


if __name__ == "__main__":
    main()
