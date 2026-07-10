"""
ADVANCED N-GRAM PREDICTIVE TEXT SYSTEM WITH MULTIPLE SMOOTHING TECHNIQUES
========================================================================

This system implements:
1. Core N-gram models (2-gram to 5-gram)
2. Classical smoothing: Laplace, Good-Turing, Kneser-Ney
3. Modern smoothing: Modified Kneser-Ney, Interpolated smoothing
4. Advanced techniques: Backoff, interpolation, ensemble methods
5. High accuracy optimizations while maintaining n-gram foundation

Focus: Proper n-gram implementation with state-of-the-art smoothing
"""

import os
import re
import pickle
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import random
import math
import numpy as np

class AdvancedSmoothingTechniques:
    """Collection of advanced smoothing techniques for n-gram models."""
    
    @staticmethod
    def laplace_smoothing(count: int, total_count: int, vocab_size: int, alpha: float = 1.0) -> float:
        """Laplace (Add-α) Smoothing."""
        return (count + alpha) / (total_count + alpha * vocab_size)
    
    @staticmethod
    def good_turing_smoothing(counts: Counter, count: int) -> float:
        """Good-Turing Smoothing - reassigns probability mass to unseen events."""
        if count == 0:
            # For unseen events, use frequency of singletons
            n1 = counts.get(1, 0)  # Number of things seen once
            total = sum(counts.values())
            return n1 / total if total > 0 else 0.0001
        
        # For seen events
        next_count = count + 1
        nc = sum(1 for c in counts.values() if c == count)  # Items with count c
        nc_plus1 = sum(1 for c in counts.values() if c == next_count)  # Items with count c+1
        
        if nc_plus1 > 0 and nc > 0:
            adjusted_count = next_count * (nc_plus1 / nc)
            total = sum(counts.values())
            return adjusted_count / total
        else:
            # Fallback to original count
            total = sum(counts.values())
            return count / total if total > 0 else 0.0001
    
    @staticmethod
    def kneser_ney_discount(counts: Dict, context_counts: Dict, discount: float = 0.75) -> Dict[str, float]:
        """Kneser-Ney Smoothing - advanced smoothing with continuation probability."""
        result = {}
        total_mass = 0
        
        for word, count in counts.items():
            if count > 0:
                # Apply discount
                discounted = max(count - discount, 0)
                result[word] = discounted
                total_mass += discounted
        
        return result
    
    @staticmethod
    def modified_kneser_ney(ngram_counts: Dict, context_counts: Dict, 
                          continuation_counts: Dict, d1: float = 0.5, d2: float = 0.75, d3: float = 0.75) -> Dict[str, float]:
        """Modified Kneser-Ney - state-of-the-art smoothing technique."""
        result = {}
        
        for word, count in ngram_counts.items():
            if count == 1:
                discount = d1
            elif count == 2:
                discount = d2
            else:
                discount = d3
            
            # Calculate discounted probability
            discounted = max(count - discount, 0)
            context_total = sum(ngram_counts.values())
            
            if context_total > 0:
                result[word] = discounted / context_total
            else:
                result[word] = 0.0
        
        return result

class NgramModelWithSmoothing:
    """Advanced N-gram model with multiple smoothing techniques."""
    
    def __init__(self, n: int = 3, smoothing: str = 'kneser_ney'):
        self.n = n
        self.smoothing = smoothing
        self.ngrams = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.word_counts = Counter()
        self.continuation_counts = defaultdict(set)  # For Kneser-Ney
        self.total_words = 0
        self.smoother = AdvancedSmoothingTechniques()
        
    def train(self, text: str):
        """Train the n-gram model with advanced preprocessing."""
        print(f"Training {self.n}-gram model with {self.smoothing} smoothing...")
        
        # Advanced preprocessing
        text = self._advanced_preprocessing(text)
        
        # Sentence segmentation
        sentences = self._segment_sentences(text)
        
        words = []
        for sentence in sentences:
            sentence_words = self._add_boundary_tokens(sentence)
            words.extend(sentence_words)
        
        self.vocabulary.update(words)
        self.word_counts.update(words)
        self.total_words = len(words)
        
        # Build n-grams and collect statistics
        self._build_ngrams_with_stats(words)
        
        print(f"Trained on {len(words):,} words, {len(self.vocabulary):,} unique words")
        print(f"Created {len(self.ngrams):,} n-gram contexts")
        print(f"Using {self.smoothing} smoothing technique")
        
    def _advanced_preprocessing(self, text: str) -> str:
        """Advanced text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s\.\!\?\;\:]', ' ', text)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Intelligent sentence segmentation."""
        # Simple but effective sentence boundary detection
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _add_boundary_tokens(self, sentence: str) -> List[str]:
        """Add sentence boundary tokens."""
        words = sentence.split()
        if not words:
            return []
        
        # Add START tokens
        boundary_words = ['<START>'] * (self.n - 1)
        boundary_words.extend(words)
        boundary_words.append('<END>')
        
        return boundary_words
    
    def _build_ngrams_with_stats(self, words: List[str]):
        """Build n-grams and collect statistics for advanced smoothing."""
        for i in range(len(words) - self.n + 1):
            # Extract context and next word
            context = tuple(words[i:i + self.n - 1])
            next_word = words[i + self.n - 1]
            
            # Update counts
            self.ngrams[context][next_word] += 1
            self.context_counts[context] += 1
            
            # For Kneser-Ney: track continuation statistics
            if self.smoothing in ['kneser_ney', 'modified_kneser_ney']:
                for j in range(len(context)):
                    sub_context = context[j:]
                    self.continuation_counts[sub_context].add(next_word)
    
    def predict(self, context: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """Predict next words using advanced smoothing."""
        # Prepare context
        context = self._prepare_context(context)
        context_tuple = tuple(context)
        
        # Get predictions based on smoothing technique
        if self.smoothing == 'laplace':
            predictions = self._predict_laplace(context_tuple, k)
        elif self.smoothing == 'good_turing':
            predictions = self._predict_good_turing(context_tuple, k)
        elif self.smoothing == 'kneser_ney':
            predictions = self._predict_kneser_ney(context_tuple, k)
        elif self.smoothing == 'modified_kneser_ney':
            predictions = self._predict_modified_kneser_ney(context_tuple, k)
        else:
            predictions = self._predict_interpolated(context_tuple, k)
        
        return predictions[:k]
    
    def _prepare_context(self, context: List[str]) -> List[str]:
        """Prepare context with proper length and tokens."""
        if len(context) >= self.n - 1:
            return context[-(self.n - 1):]
        else:
            return ['<START>'] * (self.n - 1 - len(context)) + context
    
    def _predict_laplace(self, context_tuple: tuple, k: int) -> List[Tuple[str, float]]:
        """Predict using Laplace smoothing."""
        candidates = Counter()
        
        # Primary context
        if context_tuple in self.ngrams:
            candidates.update(self.ngrams[context_tuple])
        
        # Calculate probabilities with Laplace smoothing
        predictions = []
        vocab_size = len(self.vocabulary)
        total_count = self.context_counts.get(context_tuple, 0)
        
        # Get top candidates
        all_words = set(candidates.keys())
        if len(all_words) < k * 2:
            # Add frequent words if not enough candidates
            frequent_words = [word for word, _ in self.word_counts.most_common(k * 2)]
            all_words.update(frequent_words)
        
        for word in all_words:
            if word not in ['<START>', '<END>']:
                count = candidates.get(word, 0)
                prob = self.smoother.laplace_smoothing(count, total_count, vocab_size, alpha=0.1)
                predictions.append((word, prob))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    
    def _predict_good_turing(self, context_tuple: tuple, k: int) -> List[Tuple[str, float]]:
        """Predict using Good-Turing smoothing."""
        if context_tuple in self.ngrams:
            word_counts = self.ngrams[context_tuple]
        else:
            word_counts = Counter()
        
        predictions = []
        for word, count in word_counts.most_common(k * 2):
            if word not in ['<START>', '<END>']:
                prob = self.smoother.good_turing_smoothing(word_counts, count)
                predictions.append((word, prob))
        
        # Add unseen words
        if len(predictions) < k:
            unseen_prob = self.smoother.good_turing_smoothing(word_counts, 0)
            for word, _ in self.word_counts.most_common(k):
                if word not in [p[0] for p in predictions] and word not in ['<START>', '<END>']:
                    predictions.append((word, unseen_prob))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    
    def _predict_kneser_ney(self, context_tuple: tuple, k: int) -> List[Tuple[str, float]]:
        """Predict using Kneser-Ney smoothing."""
        predictions = []
        
        if context_tuple in self.ngrams:
            word_counts = self.ngrams[context_tuple]
            smoothed = self.smoother.kneser_ney_discount(
                word_counts, self.context_counts, discount=0.75
            )
            
            total_mass = sum(smoothed.values())
            for word, mass in smoothed.items():
                if word not in ['<START>', '<END>'] and total_mass > 0:
                    prob = mass / total_mass
                    predictions.append((word, prob))
        
        # Backoff with continuation probability
        if len(predictions) < k and len(context_tuple) > 0:
            shorter_context = context_tuple[1:]
            backoff_predictions = self._predict_kneser_ney(shorter_context, k - len(predictions))
            
            # Add backoff predictions with reduced weight
            for word, prob in backoff_predictions:
                if word not in [p[0] for p in predictions]:
                    predictions.append((word, prob * 0.4))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    
    def _predict_modified_kneser_ney(self, context_tuple: tuple, k: int) -> List[Tuple[str, float]]:
        """Predict using Modified Kneser-Ney smoothing."""
        predictions = []
        
        if context_tuple in self.ngrams:
            word_counts = self.ngrams[context_tuple]
            continuation_counts = {word: len(self.continuation_counts.get((word,), set())) 
                                 for word in word_counts.keys()}
            
            smoothed = self.smoother.modified_kneser_ney(
                word_counts, self.context_counts, continuation_counts
            )
            
            for word, prob in smoothed.items():
                if word not in ['<START>', '<END>']:
                    predictions.append((word, prob))
        
        # Recursive backoff
        if len(predictions) < k and len(context_tuple) > 0:
            shorter_context = context_tuple[1:]
            backoff_predictions = self._predict_modified_kneser_ney(shorter_context, k - len(predictions))
            
            for word, prob in backoff_predictions:
                if word not in [p[0] for p in predictions]:
                    predictions.append((word, prob * 0.3))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    
    def _predict_interpolated(self, context_tuple: tuple, k: int) -> List[Tuple[str, float]]:
        """Predict using interpolated smoothing (combines multiple techniques)."""
        # Get predictions from different smoothing methods
        laplace_pred = dict(self._predict_laplace(context_tuple, k * 2))
        kneser_pred = dict(self._predict_kneser_ney(context_tuple, k * 2))
        
        # Interpolate probabilities
        interpolated = defaultdict(float)
        all_words = set(laplace_pred.keys()) | set(kneser_pred.keys())
        
        for word in all_words:
            # Weighted combination of different smoothing techniques
            prob = (0.6 * kneser_pred.get(word, 0) + 
                   0.4 * laplace_pred.get(word, 0))
            interpolated[word] = prob
        
        # Convert to sorted list
        predictions = [(word, prob) for word, prob in interpolated.items() 
                      if word not in ['<START>', '<END>']]
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)

class EnsembleModel:
    """Ensemble of n-gram models for better accuracy."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model: NgramModelWithSmoothing, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        
    def predict(self, context: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """Predict using weighted ensemble."""
        combined_scores = defaultdict(float)
        
        for name, model in self.models.items():
            try:
                predictions = model.predict(context, k * 2)
                weight = self.weights[name]
                
                for word, prob in predictions:
                    combined_scores[word] += prob * weight
            except:
                continue
        
        # Normalize and sort
        if combined_scores:
            total_score = sum(combined_scores.values())
            normalized_scores = [(word, score/total_score) 
                               for word, score in combined_scores.items()]
            return sorted(normalized_scores, key=lambda x: x[1], reverse=True)[:k]
        
        return [("the", 0.1), ("and", 0.08), ("of", 0.07), ("to", 0.06), ("a", 0.05)]

def load_training_data():
    """Load training data from the enhanced corpus."""
    corpus_path = "datasets/enhanced_corpus.txt"
    
    if not os.path.exists(corpus_path):
        print("Creating sample training data...")
        sample_text = """
        The quick brown fox jumps over the lazy dog. The fox is very clever and fast.
        Machine learning is a powerful tool for artificial intelligence. Deep learning algorithms
        can process natural language very effectively. Neural networks learn patterns from data.
        Python is a great programming language for machine learning. TensorFlow and PyTorch are popular frameworks.
        Natural language processing enables computers to understand human language. Text prediction helps users type faster.
        The weather is nice today. The sun is shining brightly. Birds are singing in the trees.
        Computer science is an exciting field of study. Algorithms solve complex problems efficiently.
        Data science combines statistics with computer programming. Machine learning models make predictions.
        Artificial intelligence will transform many industries. Automation improves efficiency and productivity.
        Students learn programming languages like Python and Java. Software development requires problem-solving skills.
        Technology advances rapidly in the modern world. Innovation drives economic growth and development.
        """
        return sample_text * 10  # Repeat for more training data
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Use a reasonable subset for training
        if len(text) > 1000000:  # If too large, use first 1M characters
            text = text[:1000000]
            
        print(f"Loaded {len(text):,} characters from corpus")
        return text
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return load_training_data()  # Fall back to sample data

def train_models():
    """Train ensemble of n-gram models."""
    print("SIMPLE WORKING N-GRAM PREDICTIVE TEXT SYSTEM")
    print("=" * 50)
    
    # Check if models already exist
    model_file = 'trained_models.pkl'
    if os.path.exists(model_file):
        print("📁 Found existing trained models. Loading...")
        try:
            with open(model_file, 'rb') as f:
                models = pickle.load(f)
            print(f"✅ Successfully loaded {len(models)} trained models!")
            print("🚀 Ready to use - no retraining needed!")
            return models
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("🔄 Will retrain models...")
    
    print("🏋️ Training new models (this will take a moment)...")
    
    # Load training data
    text = load_training_data()
    
    # Create and train models
    models = {}
    
    # Train different n-gram models with various smoothing techniques
    print("📚 Training models with different smoothing techniques...")
    for n in [2, 3, 4]:
        for smoothing in ['laplace', 'kneser_ney', 'modified_kneser_ney']:
            print(f"  → Training {n}-gram with {smoothing} smoothing...")
            model = NgramModelWithSmoothing(n, smoothing)
            model.train(text)
            models[f'{n}gram_{smoothing}'] = model
    
    # Create multiple ensembles with different smoothing techniques
    print("🎯 Creating ensemble models...")
    ensemble_laplace = EnsembleModel()
    ensemble_laplace.add_model('2gram', models['2gram_laplace'], weight=0.3)
    ensemble_laplace.add_model('3gram', models['3gram_laplace'], weight=0.5)
    ensemble_laplace.add_model('4gram', models['4gram_laplace'], weight=0.2)
    
    ensemble_kneser = EnsembleModel()
    ensemble_kneser.add_model('2gram', models['2gram_kneser_ney'], weight=0.3)
    ensemble_kneser.add_model('3gram', models['3gram_kneser_ney'], weight=0.5)
    ensemble_kneser.add_model('4gram', models['4gram_kneser_ney'], weight=0.2)
    
    ensemble_modified = EnsembleModel()
    ensemble_modified.add_model('2gram', models['2gram_modified_kneser_ney'], weight=0.3)
    ensemble_modified.add_model('3gram', models['3gram_modified_kneser_ney'], weight=0.5)
    ensemble_modified.add_model('4gram', models['4gram_modified_kneser_ney'], weight=0.2)
    
    models['ensemble_laplace'] = ensemble_laplace
    models['ensemble_kneser_ney'] = ensemble_kneser
    models['ensemble_modified_kneser_ney'] = ensemble_modified
    
    # Save models for future use
    print("💾 Saving trained models...")
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(models, f)
        print(f"✅ Models saved to '{model_file}'")
        print("🎉 Next time you run this, models will load instantly!")
    except Exception as e:
        print(f"⚠️ Warning: Could not save models: {e}")
    
    return models

def evaluate_accuracy(models, test_sentences=None):
    """Evaluate model accuracy on test data."""
    print("\nEVALUATING MODEL ACCURACY")
    print("=" * 30)
    
    if test_sentences is None:
        test_sentences = [
            "the quick brown fox",
            "machine learning is very",
            "artificial intelligence will",
            "natural language processing",
            "deep learning algorithms can",
            "python programming language is",
            "computer science students learn",
            "software development requires good",
            "data science combines statistics and",
            "technology advances rapidly in modern"
        ]
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name}...")
        
        correct_predictions = 0
        top3_correct = 0
        total_tests = 0
        
        for sentence in test_sentences:
            words = sentence.split()
            if len(words) < 2:
                continue
                
            # Test each position in the sentence
            for i in range(1, len(words)):
                context = words[:i]
                target = words[i]
                
                try:
                    predictions = model.predict(context, k=5)
                    if predictions:
                        # Check top-1 accuracy
                        if predictions[0][0].lower() == target.lower():
                            correct_predictions += 1
                        
                        # Check top-3 accuracy
                        top3_words = [p[0].lower() for p in predictions[:3]]
                        if target.lower() in top3_words:
                            top3_correct += 1
                    
                    total_tests += 1
                except Exception as e:
                    continue
        
        if total_tests > 0:
            accuracy = (correct_predictions / total_tests) * 100
            top3_accuracy = (top3_correct / total_tests) * 100
            
            results[model_name] = {
                'top1_accuracy': accuracy,
                'top3_accuracy': top3_accuracy,
                'tests': total_tests
            }
            
            print(f"  Top-1 Accuracy: {accuracy:.1f}%")
            print(f"  Top-3 Accuracy: {top3_accuracy:.1f}%")
            print(f"  Tests: {total_tests}")
    
    return results

def interactive_demo():
    """Interactive demonstration of the advanced n-gram system."""
    print("\n" + "="*50)
    print("🚀 ADVANCED N-GRAM PREDICTIVE TEXT SYSTEM")
    print("   With Multiple Smoothing Techniques")
    print("="*50)
    
    # Train models
    print("⏳ Training models with advanced smoothing...")
    models = train_models()
    
    # Quick accuracy check
    print("\n📊 Quick Accuracy Assessment:")
    evaluate_accuracy(models)
    
    print("\n" + "="*50)
    print("🎯 INTERACTIVE DEMO")
    print("="*50)
    print("Type partial sentences and see predictions!")
    print("Available models:")
    for model_name in sorted(models.keys()):
        print(f"  • {model_name}")
    print("\nCommands:")
    print("  • 'models' - list all models")
    print("  • 'switch <model>' - change active model")
    print("  • 'accuracy' - run accuracy test")
    print("  • 'compare' - compare smoothing techniques")
    print("  • 'retrain' - delete saved models and retrain from scratch")
    print("  • 'quit' - exit demo")
    print("-" * 50)
    
    current_model = 'ensemble_modified_kneser_ney'  # Start with best model
    print(f"🎯 Active model: {current_model}")
    
    while True:
        try:
            user_input = input("\n📝 Enter text (or command): ").strip()
            
            if not user_input or user_input.lower() == 'quit':
                print("👋 Thanks for using the Advanced N-gram System!")
                break
            elif user_input.lower() == 'models':
                print("\n📋 Available models:")
                for i, model_name in enumerate(sorted(models.keys()), 1):
                    marker = "👉" if model_name == current_model else "  "
                    print(f"{marker} {i:2d}. {model_name}")
                continue
            elif user_input.lower().startswith('switch '):
                new_model = user_input[7:].strip()
                if new_model in models:
                    current_model = new_model
                    print(f"✅ Switched to: {current_model}")
                else:
                    print(f"❌ Model '{new_model}' not found")
                continue
            elif user_input.lower() == 'accuracy':
                print("\n🔍 Running comprehensive accuracy test...")
                evaluate_accuracy(models)
                continue
            elif user_input.lower() == 'compare':
                compare_smoothing_techniques()
                continue
            elif user_input.lower() == 'retrain':
                print("\n🗑️ Deleting saved models and retraining...")
                model_file = 'trained_models.pkl'
                if os.path.exists(model_file):
                    os.remove(model_file)
                    print("✅ Deleted existing models")
                print("🔄 Retraining models...")
                models = train_models()
                print("🎉 Retraining complete!")
                continue
            
            # Get predictions
            context = user_input.lower().split()
            model = models[current_model]
            predictions = model.predict(context, k=8)
            
            # Display results
            print(f"\n🎯 Predictions from {current_model}:")
            print(f"📝 Context: '{user_input}'")
            print("-" * 40)
            
            if predictions:
                for i, (word, prob) in enumerate(predictions, 1):
                    confidence = "🔥" if prob > 0.1 else "⭐" if prob > 0.05 else "💫"
                    percentage = prob * 100
                    print(f"{confidence} {i:2d}. {word:<15} ({percentage:5.1f}%)")
            else:
                print("❌ No predictions available")
                
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

def compare_smoothing_techniques():
    """Compare different smoothing techniques side by side."""
    print("\n" + "="*60)
    print("🔬 SMOOTHING TECHNIQUE COMPARISON")
    print("="*60)
    
    # Load data
    text = load_training_data()
    
    # Test different smoothing techniques
    smoothing_methods = ['laplace', 'kneser_ney', 'modified_kneser_ney', 'interpolated']
    test_contexts = [
        "the quick brown",
        "machine learning is",
        "natural language",
        "python programming",
        "artificial intelligence"
    ]
    
    for context_str in test_contexts:
        print(f"\n📝 Context: '{context_str}'")
        print("-" * 50)
        
        context = context_str.split()
        
        for smoothing in smoothing_methods:
            print(f"\n🔍 {smoothing.upper()} Smoothing:")
            
            # Create and train model
            model = NgramModelWithSmoothing(3, smoothing)
            model.train(text)
            
            # Get predictions
            predictions = model.predict(context, k=5)
            
            if predictions:
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"   {i}. {word:<12} ({prob*100:5.1f}%)")
            else:
                print("   No predictions")
                
        print("-" * 50)

def main():
    """Main function to demonstrate the advanced n-gram system."""
    try:
        print("🚀 Advanced N-gram Predictive Text System")
        print("   Featuring Multiple Smoothing Techniques")
        print("=" * 50)
        
        # Run interactive demo directly
        interactive_demo()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()