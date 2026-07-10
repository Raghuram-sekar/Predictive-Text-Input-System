"""
Training Example

This script demonstrates different ways to train n-gram models
including various data sources and training configurations.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ngram_model import NgramModel
from src.predictor import Predictor
from src.evaluator import ModelEvaluator
from src.smoothing import SmoothedNgramModel


def train_from_text_example():
    """Train model from raw text string."""
    print("=== Training from Text String ===\n")
    
    # Sample text for training
    sample_text = """
    Natural language processing is a subfield of artificial intelligence that focuses on 
    the interaction between computers and humans through natural language. The ultimate 
    goal of NLP is to enable computers to understand, interpret, and generate human language 
    in a valuable way.
    
    Machine learning algorithms are at the core of modern NLP systems. These algorithms 
    can learn patterns from large amounts of text data and use these patterns to make 
    predictions about new, unseen text.
    
    Some common NLP tasks include text classification, sentiment analysis, named entity 
    recognition, machine translation, and question answering. Each of these tasks requires 
    different approaches and techniques.
    """
    
    # Train bigram model
    print("Training bigram model from text...")
    bigram_model = NgramModel(n=2, smoothing_alpha=0.01)
    bigram_model.train_from_text(sample_text)
    
    print(f"Model trained successfully!")
    print(f"Vocabulary size: {bigram_model.vocab_size}")
    print(f"Total bigrams: {bigram_model.total_ngrams}")
    
    # Test predictions
    predictor = Predictor(bigram_model)
    test_context = "natural language"
    predictions = predictor.predict(test_context, top_k=5)
    print(f"Predictions for '{test_context}': {predictions}")
    
    return bigram_model


def train_from_file_example():
    """Train model from text file."""
    print("\n=== Training from File ===\n")
    
    corpus_path = "data/sample_corpus.txt"
    
    if not os.path.exists(corpus_path):
        print(f"Sample corpus not found at {corpus_path}")
        print("Creating a sample file...")
        
        # Create sample file
        os.makedirs("data", exist_ok=True)
        sample_content = """
        Artificial intelligence and machine learning are transforming industries worldwide.
        Deep learning neural networks can process complex patterns in data.
        Natural language processing enables computers to understand human language.
        Computer vision systems can analyze and interpret visual information.
        Robotics combines AI with mechanical engineering for autonomous systems.
        """
        
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print(f"Created sample corpus at {corpus_path}")
    
    # Train trigram model
    print("Training trigram model from file...")
    trigram_model = NgramModel(n=3, smoothing_alpha=0.01)
    trigram_model.train_from_file(corpus_path)
    
    print(f"Model trained successfully!")
    print(f"Vocabulary size: {trigram_model.vocab_size}")
    print(f"Total trigrams: {trigram_model.total_ngrams}")
    
    # Test predictions
    predictor = Predictor(trigram_model)
    test_context = "artificial intelligence and"
    predictions = predictor.predict(test_context, top_k=5)
    print(f"Predictions for '{test_context}': {predictions}")
    
    return trigram_model


def train_from_corpus_example():
    """Train model from multiple text documents."""
    print("\n=== Training from Multiple Documents ===\n")
    
    # Create multiple documents
    documents = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of a broader family of machine learning methods based on neural networks.",
        "Natural language processing combines computational linguistics with statistical and machine learning models.",
        "Computer vision is an interdisciplinary field that deals with how computers can be made to gain understanding from digital images.",
        "Artificial intelligence research has been highly successful in developing effective techniques for solving problems."
    ]
    
    print(f"Training on {len(documents)} documents...")
    
    # Train model
    model = NgramModel(n=2, smoothing_alpha=0.05)
    model.train_from_corpus(documents)
    
    print(f"Model trained successfully!")
    print(f"Vocabulary size: {model.vocab_size}")
    print(f"Total bigrams: {model.total_ngrams}")
    
    # Show most frequent bigrams
    print("\nMost frequent bigrams:")
    sorted_bigrams = sorted(model.ngram_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (bigram, count) in enumerate(sorted_bigrams[:10]):
        bigram_str = " ".join(bigram)
        print(f"  {i+1:2d}. {bigram_str}: {count}")
    
    return model


def train_with_different_parameters():
    """Demonstrate training with different parameters."""
    print("\n=== Training with Different Parameters ===\n")
    
    sample_text = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping under 
    the tree when the fox appeared. The fox was looking for food in the forest.
    The forest was quiet except for the sound of birds singing in the trees.
    """
    
    # Different smoothing values
    smoothing_values = [0.001, 0.01, 0.1, 1.0]
    models = {}
    
    print("Training models with different smoothing parameters:")
    
    for alpha in smoothing_values:
        print(f"\nTraining with alpha = {alpha}")
        model = NgramModel(n=2, smoothing_alpha=alpha)
        model.train_from_text(sample_text)
        models[alpha] = model
        
        # Test prediction
        predictor = Predictor(model)
        predictions = predictor.predict("the quick", top_k=3, include_probabilities=True)
        
        print(f"  Predictions for 'the quick':")
        for word, prob in predictions:
            print(f"    {word}: {prob:.4f}")
    
    return models


def train_and_save_model():
    """Train model and save to file."""
    print("\n=== Training and Saving Model ===\n")
    
    # Train model
    model = NgramModel(n=2, smoothing_alpha=0.01)
    model.train_from_file("data/sample_corpus.txt")
    
    # Save model
    model_path = "trained_bigram_model.pkl"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Load model back
    loaded_model = NgramModel()
    loaded_model.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Verify models are identical
    assert model.n == loaded_model.n
    assert model.vocab_size == loaded_model.vocab_size
    assert model.total_ngrams == loaded_model.total_ngrams
    print("Model save/load verification successful!")
    
    # Clean up
    os.remove(model_path)
    print("Temporary model file removed")
    
    return loaded_model


def train_with_advanced_smoothing():
    """Demonstrate training with advanced smoothing techniques."""
    print("\n=== Training with Advanced Smoothing ===\n")
    
    # Train base model
    base_model = NgramModel(n=2, smoothing_alpha=0.01)
    base_model.train_from_file("data/sample_corpus.txt")
    
    # Create smoothed models
    smoothing_methods = ['laplace', 'good_turing', 'kneser_ney']
    
    print("Comparing smoothing methods:")
    
    for method in smoothing_methods:
        print(f"\n{method.upper()} smoothing:")
        try:
            smoothed_model = SmoothedNgramModel(base_model, smoothing_method=method)
            
            # Test on unseen n-gram
            test_ngram = ('artificial', 'neural')  # Likely unseen
            prob = smoothed_model.get_smoothed_probability(test_ngram)
            print(f"  P(neural|artificial) = {prob:.6f}")
            
        except Exception as e:
            print(f"  Error with {method}: {str(e)}")


def comprehensive_training_example():
    """Comprehensive training workflow example."""
    print("\n=== Comprehensive Training Workflow ===\n")
    
    print("1. Data Preparation")
    print("   - Loading training data...")
    
    # Prepare training data
    if os.path.exists("data/sample_corpus.txt"):
        with open("data/sample_corpus.txt", 'r', encoding='utf-8') as f:
            training_text = f.read()
    else:
        training_text = """
        Machine learning is revolutionizing how we process and understand data.
        Neural networks can learn complex patterns from examples.
        Deep learning models achieve state-of-the-art performance in many tasks.
        Natural language processing enables human-computer interaction.
        """
    
    print(f"   - Training data length: {len(training_text)} characters")
    
    print("\n2. Model Training")
    print("   - Training bigram model...")
    bigram_model = NgramModel(n=2, smoothing_alpha=0.01)
    bigram_model.train_from_text(training_text)
    
    print("   - Training trigram model...")
    trigram_model = NgramModel(n=3, smoothing_alpha=0.01)
    trigram_model.train_from_text(training_text)
    
    print("\n3. Model Comparison")
    models = {'Bigram': bigram_model, 'Trigram': trigram_model}
    
    for name, model in models.items():
        stats = model.get_model_stats()
        print(f"   {name} model:")
        print(f"     Vocabulary size: {stats['vocabulary_size']}")
        print(f"     Total n-grams: {stats['total_ngrams']}")
        print(f"     Unique n-grams: {stats['unique_ngrams']}")
    
    print("\n4. Prediction Testing")
    test_context = "machine learning"
    
    for name, model in models.items():
        predictor = Predictor(model)
        predictions = predictor.predict(test_context, top_k=3)
        print(f"   {name}: '{test_context}' → {predictions}")
    
    print("\n5. Model Evaluation")
    test_sentences = [
        "machine learning algorithms are powerful",
        "neural networks learn from data",
        "deep learning models are sophisticated"
    ]
    
    for name, model in models.items():
        evaluator = ModelEvaluator(model)
        perplexity = evaluator.evaluate_perplexity(test_sentences)
        print(f"   {name} perplexity: {perplexity:.4f}")
    
    return models


def main():
    """Run all training examples."""
    print("Predictive Text Input System - Training Examples")
    print("=" * 60)
    
    try:
        train_from_text_example()
        train_from_file_example()
        train_from_corpus_example()
        train_with_different_parameters()
        train_and_save_model()
        train_with_advanced_smoothing()
        comprehensive_training_example()
        
        print("\n" + "=" * 60)
        print("All training examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running training examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
