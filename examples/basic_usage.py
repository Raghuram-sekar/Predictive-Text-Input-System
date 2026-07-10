"""
Basic Usage Examples

This script demonstrates basic usage of the Predictive Text Input System
including model training, prediction, and evaluation.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ngram_model import NgramModel
from src.predictor import Predictor
from src.evaluator import ModelEvaluator


def basic_bigram_example():
    """Demonstrate basic bigram model usage."""
    print("=== Basic Bigram Model Example ===\n")
    
    # Create and train a bigram model
    print("1. Creating and training a bigram model...")
    model = NgramModel(n=2, smoothing_alpha=0.01)
    
    # Sample training text
    training_text = """
    Artificial intelligence is transforming the world. Machine learning algorithms 
    can process vast amounts of data. Natural language processing enables computers 
    to understand human language. Deep learning models revolutionize computer vision.
    Neural networks are inspired by the human brain. The future of artificial 
    intelligence looks very promising.
    """
    
    model.train_from_text(training_text)
    print(f"   Vocabulary size: {model.vocab_size}")
    print(f"   Total bigrams: {model.total_ngrams}")
    
    # Create predictor
    print("\n2. Creating predictor...")
    predictor = Predictor(model)
    
    # Make predictions
    print("\n3. Making predictions...")
    test_contexts = [
        "artificial",
        "machine learning",
        "natural language",
        "deep learning",
        "neural networks"
    ]
    
    for context in test_contexts:
        predictions = predictor.predict(context, top_k=3)
        print(f"   Context: '{context}' → Predictions: {predictions}")
    
    # Show model statistics
    print("\n4. Model statistics:")
    stats = model.get_model_stats()
    for key, value in stats.items():
        if isinstance(value, float) and value is not None:
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


def trigram_comparison_example():
    """Compare bigram and trigram models."""
    print("\n=== Bigram vs Trigram Comparison ===\n")
    
    # Load sample corpus
    corpus_path = os.path.join("data", "sample_corpus.txt")
    if not os.path.exists(corpus_path):
        print("Sample corpus not found. Using simple text.")
        training_text = """
        Machine learning is a subset of artificial intelligence. Deep learning is 
        a subset of machine learning. Natural language processing uses machine 
        learning algorithms. Computer vision also uses deep learning models.
        """
    else:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            training_text = f.read()
    
    # Train bigram model
    print("1. Training bigram model...")
    bigram_model = NgramModel(n=2)
    bigram_model.train_from_text(training_text)
    
    # Train trigram model
    print("2. Training trigram model...")
    trigram_model = NgramModel(n=3)
    trigram_model.train_from_text(training_text)
    
    # Create predictors
    bigram_predictor = Predictor(bigram_model)
    trigram_predictor = Predictor(trigram_model)
    
    # Compare predictions
    print("\n3. Comparing predictions:")
    test_contexts = [
        "artificial intelligence",
        "machine learning algorithms",
        "natural language processing"
    ]
    
    for context in test_contexts:
        bigram_preds = bigram_predictor.predict(context, top_k=3)
        trigram_preds = trigram_predictor.predict(context, top_k=3)
        
        print(f"\n   Context: '{context}'")
        print(f"   Bigram predictions:  {bigram_preds}")
        print(f"   Trigram predictions: {trigram_preds}")


def prediction_features_example():
    """Demonstrate various prediction features."""
    print("\n=== Prediction Features Example ===\n")
    
    # Create model
    model = NgramModel(n=2)
    
    # Train with more diverse text
    training_text = """
    The weather is beautiful today. The sun is shining brightly. The sky is 
    clear and blue. The temperature is perfect for outdoor activities. The 
    birds are singing in the trees. The flowers are blooming in the garden.
    
    Technology is advancing rapidly. Computers are becoming more powerful. 
    Smartphones are getting smarter every year. The internet connects people 
    worldwide. Social media platforms enable global communication.
    
    Education is very important. Students need to study hard. Teachers help 
    students learn new concepts. Books contain valuable knowledge. Libraries 
    are treasure troves of information.
    """
    
    model.train_from_text(training_text)
    predictor = Predictor(model)
    
    print("1. Basic prediction:")
    predictions = predictor.predict("the weather", top_k=5)
    print(f"   'the weather' → {predictions}")
    
    print("\n2. Prediction with probabilities:")
    predictions_with_probs = predictor.predict("the weather", top_k=3, 
                                             include_probabilities=True)
    for word, prob in predictions_with_probs:
        print(f"   '{word}': {prob:.4f}")
    
    print("\n3. Sentence completion:")
    completion = predictor.predict_sentence_completion("the sun is", max_words=5)
    print(f"   'the sun is' → 'the sun is {completion}'")
    
    print("\n4. Word suggestions (partial input):")
    suggestions = predictor.get_word_suggestions("tec", context="modern", top_k=3)
    print(f"   'modern tec...' → {suggestions}")
    
    print("\n5. Text probability evaluation:")
    test_sentences = [
        "the weather is beautiful",
        "computers are very smart",
        "random nonsense words here"
    ]
    
    for sentence in test_sentences:
        prob = predictor.evaluate_text_probability(sentence)
        print(f"   '{sentence}': log_prob = {prob:.4f}")


def interactive_session_example():
    """Demonstrate interactive prediction session."""
    print("\n=== Interactive Session Example ===\n")
    
    # Create model
    model = NgramModel(n=2)
    model.train_from_file("data/sample_corpus.txt")
    
    predictor = Predictor(model)
    
    # Start interactive session
    from src.predictor import InteractivePredictionSession
    session = InteractivePredictionSession(predictor)
    
    print("Simulating an interactive session:")
    
    # Simulate user input
    user_inputs = [
        "artificial",
        "intelligence",
        "is",
        "transforming"
    ]
    
    for word in user_inputs:
        session.add_text(word)
        predictions = session.predict_next(top_k=3)
        session_text = session.get_session_text()
        
        print(f"   Added: '{word}'")
        print(f"   Current text: '{session_text}'")
        print(f"   Next predictions: {predictions}")
        
        # Simulate accepting first prediction
        if predictions:
            session.accept_prediction(predictions[0])
            print(f"   Accepted: '{predictions[0]}'")
        print()


def evaluation_example():
    """Demonstrate model evaluation."""
    print("\n=== Model Evaluation Example ===\n")
    
    # Train model
    model = NgramModel(n=2)
    model.train_from_file("data/sample_corpus.txt")
    
    # Load test data
    test_path = "data/test_data.txt"
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_sentences = [line.strip() for line in f if line.strip()]
    else:
        test_sentences = [
            "artificial intelligence is very powerful",
            "machine learning algorithms are sophisticated",
            "natural language processing enables understanding"
        ]
    
    # Evaluate model
    evaluator = ModelEvaluator(model)
    
    print("1. Prediction accuracy:")
    accuracy = evaluator.evaluate_prediction_accuracy(test_sentences[:5], top_k=3)
    for key, value in accuracy.items():
        if 'accuracy' in key:
            print(f"   {key}: {value:.4f} ({value*100:.2f}%)")
    
    print("\n2. Perplexity:")
    perplexity = evaluator.evaluate_perplexity(test_sentences[:5])
    print(f"   Perplexity: {perplexity:.4f}")
    
    print("\n3. Vocabulary coverage:")
    coverage = evaluator.evaluate_vocabulary_coverage(test_sentences)
    print(f"   Coverage: {coverage['coverage_ratio']:.4f} ({coverage['coverage_ratio']*100:.2f}%)")
    print(f"   OOV rate: {coverage['oov_ratio']:.4f} ({coverage['oov_ratio']*100:.2f}%)")


def main():
    """Run all examples."""
    print("Predictive Text Input System - Basic Usage Examples")
    print("=" * 60)
    
    try:
        basic_bigram_example()
        trigram_comparison_example()
        prediction_features_example()
        interactive_session_example()
        evaluation_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
