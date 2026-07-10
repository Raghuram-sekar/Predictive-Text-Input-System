"""
LIVE PREDICTIVE TEXT DEMO
========================

This file loads the trained models and provides a clean live demo interface.
"""

import pickle
import os
from typing import List, Tuple

class LivePredictor:
    """Live prediction interface."""
    
    def __init__(self):
        self.models = None
        self.ensemble = None
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        try:
            if os.path.exists('trained_models.pkl'):
                with open('trained_models.pkl', 'rb') as f:
                    self.models = pickle.load(f)
                self.ensemble = self.models.get('ensemble', self.models.get('3gram'))
                print("Models loaded successfully!")
            else:
                print("No trained models found. Please run simple_working_demo.py first.")
                return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        return True
    
    def predict(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Get predictions for text."""
        if not self.ensemble:
            return []
        
        words = text.lower().strip().split()
        return self.ensemble.predict(words, k)
    
    def format_predictions(self, predictions: List[Tuple[str, float]]) -> str:
        """Format predictions for display."""
        if not predictions:
            return "No predictions available"
        
        result = []
        for i, (word, prob) in enumerate(predictions, 1):
            result.append(f"{i}. {word:<15} ({prob:.3f})")
        
        return "\n".join(result)

def interactive_demo():
    """Interactive demo interface."""
    predictor = LivePredictor()
    
    if not predictor.models:
        return
    
    print("\n" + "="*60)
    print("             LIVE PREDICTIVE TEXT DEMO")
    print("="*60)
    print("Instructions:")
    print("- Type any text to see word predictions")
    print("- Press Enter after typing to get suggestions")
    print("- Type 'quit' or 'exit' to stop")
    print("- Type 'help' for examples")
    print("-"*60)
    
    sample_inputs = [
        "the quick brown",
        "machine learning is",
        "artificial intelligence",
        "natural language",
        "computer science",
        "python programming"
    ]
    
    while True:
        try:
            user_input = input("\n📝 Enter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the predictive text system!")
                break
            
            if user_input.lower() == 'help':
                print("\n💡 Try these examples:")
                for i, example in enumerate(sample_inputs, 1):
                    print(f"   {i}. {example}")
                continue
            
            if not user_input:
                print("⚠️  Please enter some text")
                continue
            
            # Get predictions
            predictions = predictor.predict(user_input, k=5)
            
            print(f"\n🔮 Predictions for '{user_input}':")
            print("-" * 40)
            
            formatted = predictor.format_predictions(predictions)
            print(formatted)
            
            # Show confidence
            if predictions:
                best_confidence = predictions[0][1]
                if best_confidence > 0.1:
                    print(f"\n✅ High confidence: {best_confidence:.3f}")
                elif best_confidence > 0.05:
                    print(f"\n⚠️  Medium confidence: {best_confidence:.3f}")
                else:
                    print(f"\n❓ Low confidence: {best_confidence:.3f}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def batch_demo():
    """Demonstrate predictions on batch of examples."""
    predictor = LivePredictor()
    
    if not predictor.models:
        return
    
    test_cases = [
        "the quick brown",
        "machine learning is very",
        "artificial intelligence will help",
        "natural language processing enables",
        "deep learning algorithms can",
        "python programming language",
        "computer science students",
        "software development requires",
        "data science combines",
        "technology advances rapidly"
    ]
    
    print("\n" + "="*60)
    print("             BATCH PREDICTION DEMO")
    print("="*60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. Input: '{test_input}'")
        predictions = predictor.predict(test_input, k=3)
        
        if predictions:
            next_words = [pred[0] for pred in predictions]
            print(f"   Predictions: {', '.join(next_words)}")
        else:
            print("   No predictions")

def accuracy_demo():
    """Show accuracy on known phrases."""
    predictor = LivePredictor()
    
    if not predictor.models:
        return
    
    # Test phrases where we know the next word
    test_phrases = [
        ("artificial intelligence", "will"),
        ("machine learning", "is"),
        ("natural language", "processing"),
        ("computer science", "students"),
        ("programming language", "python"),
        ("data science", "combines"),
        ("deep learning", "algorithms"),
        ("software development", "requires")
    ]
    
    print("\n" + "="*60)
    print("             ACCURACY DEMONSTRATION")
    print("="*60)
    
    correct = 0
    total = len(test_phrases)
    
    for context, expected in test_phrases:
        predictions = predictor.predict(context, k=5)
        
        if predictions:
            predicted_words = [pred[0].lower() for pred in predictions]
            is_correct = expected.lower() in predicted_words
            position = predicted_words.index(expected.lower()) + 1 if is_correct else "Not found"
            
            status = "✅" if is_correct else "❌"
            print(f"{status} '{context}' → Expected: '{expected}', Position: {position}")
            
            if is_correct:
                correct += 1
        else:
            print(f"❌ '{context}' → No predictions")
    
    accuracy = (correct / total) * 100
    print(f"\n📊 Accuracy: {correct}/{total} = {accuracy:.1f}%")

def main():
    """Main demo menu."""
    print("\n🤖 PREDICTIVE TEXT SYSTEM - LIVE DEMO")
    print("="*50)
    print("Choose demo mode:")
    print("1. Interactive Demo (recommended)")
    print("2. Batch Demo")
    print("3. Accuracy Demo")
    print("4. All Demos")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                interactive_demo()
                break
            elif choice == '2':
                batch_demo()
                break
            elif choice == '3':
                accuracy_demo()
                break
            elif choice == '4':
                batch_demo()
                accuracy_demo()
                interactive_demo()
                break
            else:
                print("❌ Please enter 1, 2, 3, or 4")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()