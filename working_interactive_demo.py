"""
Working Interactive Typing Demo

This creates a real-time typing interface that actually works with our trained model.
It handles the context properly and shows meaningful predictions.
"""

import sys
import os
from legitimate_accuracy_improvement import (
    ImprovedNgramModel, SmartPreprocessor, create_diverse_training_data, 
    proper_train_test_split
)

class RealTimePredictor:
    """Real-time predictor that works properly with our model."""
    
    def __init__(self):
        self.preprocessor = SmartPreprocessor()
        self.model = None
        self.current_text = ""
        self.current_tokens = []
        
    def train_model(self):
        """Train our model with the same data we used before."""
        print("Training model with our legitimate data...")
        
        # Use the same training approach
        diverse_data = create_diverse_training_data()
        train_data, _ = proper_train_test_split(diverse_data, test_ratio=0.2)
        
        # Prepare training tokens
        all_tokens = []
        for sentence in train_data:
            tokens = self.preprocessor.preprocess_text(sentence)
            all_tokens.extend(tokens)
        
        # Train the model
        self.model = ImprovedNgramModel(n=3, smoothing_alpha=0.05, 
                                       vocab_threshold=1, use_backoff=True)
        self.model.train_from_tokens(all_tokens)
        
        print(f"Model trained!")
        print(f"   Vocabulary size: {len(self.model.vocabulary)}")
        print(f"   Total n-grams: {len(self.model.ngram_counts)}")
        
        # Show some example patterns the model learned
        print(f"\nSome patterns the model learned:")
        example_ngrams = list(self.model.ngram_counts.items())[:10]
        for ngram, count in example_ngrams:
            if len(ngram) == 3 and count > 1:
                print(f"   {ngram[0]} {ngram[1]} → {ngram[2]} (count: {count})")
    
    def update_text(self, new_word):
        """Update current text with new word."""
        if self.current_text:
            self.current_text += " " + new_word
        else:
            self.current_text = new_word
        
        # Update tokens
        self.current_tokens = self.preprocessor.preprocess_text(self.current_text)
        
    def get_predictions(self, top_k=5):
        """Get predictions for next word."""
        if not self.model or len(self.current_tokens) < 2:
            return []
        
        # Get context (last 2 words, excluding sentence boundaries)
        valid_tokens = [t for t in self.current_tokens if t not in ['<START>', '<END>']]
        
        if len(valid_tokens) < 2:
            return []
        
        context = tuple(valid_tokens[-2:])  # Last 2 words
        
        try:
            predictions = self.model.predict_next_words(context, top_k=top_k)
            # Filter out sentence boundary tokens for display
            filtered_predictions = [
                (word, prob) for word, prob in predictions 
                if word not in ['<START>', '<END>', '<UNK>']
            ]
            return filtered_predictions
        except Exception as e:
            print(f"Prediction error: {e}")
            return []
    
    def get_context_info(self):
        """Get information about current context."""
        valid_tokens = [t for t in self.current_tokens if t not in ['<START>', '<END>']]
        
        if len(valid_tokens) >= 2:
            context = tuple(valid_tokens[-2:])
            # Check if this context exists in our model
            context_exists = any(ngram[:2] == context for ngram in self.model.ngram_counts.keys())
            return context, context_exists
        
        return None, False


def run_interactive_demo():
    """Run the interactive typing demo."""
    print("REAL-TIME TYPING DEMO")
    print("=" * 50)
    print("Type words one by one and see predictions!")
    print("Commands: 'quit' to exit, 'reset' to start over, 'help' for help")
    print()
    
    predictor = RealTimePredictor()
    predictor.train_model()
    
    print("\n" + "=" * 50)
    print("Ready! Start typing...")
    print("=" * 50)
    
    while True:
        try:
            # Show current text
            print(f"\nCurrent text: '{predictor.current_text}'")
            
            # Show context info
            context, context_exists = predictor.get_context_info()
            if context:
                status = "Known" if context_exists else "Unknown"
                print(f"Context: {context} ({status})")
            
            # Get and show predictions
            predictions = predictor.get_predictions(top_k=5)
            if predictions:
                print("Predictions:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"   {i}. {word} ({prob:.3f})")
            else:
                print("No predictions available")
            
            # Get next word from user
            print()
            next_word = input("Type next word: ").strip().lower()
            
            # Handle commands
            if next_word == 'quit':
                print("Goodbye!")
                break
            elif next_word == 'reset':
                predictor.current_text = ""
                predictor.current_tokens = []
                print("Reset! Starting fresh...")
                continue
            elif next_word == 'help':
                print("\nHELP:")
                print("   - Type words one by one to build a sentence")
                print("   - The model will predict the next word after each input")
                print("   - Try words from training data like: the, cat, sat, on, mat")
                print("   - Or: alice, fell, down, rabbit, hole")
                print("   - Or: sherlock, holmes, solved, mystery")
                print("   - 'reset' to start over, 'quit' to exit")
                continue
            elif not next_word:
                print("Please enter a word!")
                continue
            
            # Update text and continue
            predictor.update_text(next_word)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def test_known_patterns():
    """Test with patterns we know the model learned."""
    print("TESTING KNOWN PATTERNS")
    print("=" * 50)
    
    predictor = RealTimePredictor()
    predictor.train_model()
    
    # Test patterns we know should work
    test_patterns = [
        ["the", "cat"],
        ["alice", "fell"],
        ["sherlock", "holmes"],
        ["the", "detective"],
        ["once", "upon"]
    ]
    
    for pattern in test_patterns:
        predictor.current_text = " ".join(pattern)
        predictor.current_tokens = predictor.preprocessor.preprocess_text(predictor.current_text)
        
        context, context_exists = predictor.get_context_info()
        predictions = predictor.get_predictions(top_k=3)
        
        print(f"\nInput: '{' '.join(pattern)}'")
        print(f"Context: {context} ({'Known' if context_exists else 'Unknown'})")
        if predictions:
            print("Predictions:")
            for word, prob in predictions:
                print(f"   → {word} ({prob:.3f})")
        else:
            print("   No predictions")


def main():
    """Main function with menu."""
    print("INTERACTIVE MODEL TESTING")
    print("=" * 50)
    print("Choose an option:")
    print("1. Interactive typing demo")
    print("2. Test known patterns first")
    print("3. Both (test patterns then interactive)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            run_interactive_demo()
        elif choice == "2":
            test_known_patterns()
        elif choice == "3":
            test_known_patterns()
            input("\nPress Enter to start interactive demo...")
            run_interactive_demo()
        else:
            print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
