"""
Interactive User Interface Module

This module provides a command-line interface for real-time text prediction
using the trained n-gram models.
"""

import os
import sys
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ngram_model import NgramModel
from src.predictor import Predictor, InteractivePredictionSession
from src.evaluator import ModelEvaluator


class InteractiveUI:
    """
    Interactive command-line interface for the Predictive Text System.
    """
    
    def __init__(self):
        """Initialize the interactive UI."""
        self.model = None
        self.predictor = None
        self.session = None
        self.running = True
    
    def display_banner(self):
        """Display the application banner."""
        print("=" * 60)
        print("  PREDICTIVE TEXT INPUT SYSTEM")
        print("  N-gram Based Markov Models for Text Prediction")
        print("=" * 60)
        print()
    
    def display_menu(self):
        """Display the main menu."""
        print("\nMAIN MENU:")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Interactive prediction")
        print("4. Batch prediction")
        print("5. Model evaluation")
        print("6. Model statistics")
        print("7. Help")
        print("8. Exit")
        print("-" * 30)
    
    def get_user_choice(self) -> str:
        """Get user menu choice."""
        try:
            choice = input("Enter your choice (1-8): ").strip()
            return choice
        except (EOFError, KeyboardInterrupt):
            return "8"  # Exit on Ctrl+C or EOF
    
    def train_new_model(self):
        """Train a new n-gram model."""
        print("\n--- TRAIN NEW MODEL ---")
        
        # Get model parameters
        try:
            n = int(input("Enter n-gram order (2 for bigram, 3 for trigram): ").strip())
            if n < 1:
                print("Error: N-gram order must be at least 1")
                return
        except ValueError:
            print("Error: Invalid n-gram order")
            return
        
        alpha = input("Enter smoothing alpha (default 0.01): ").strip()
        try:
            alpha = float(alpha) if alpha else 0.01
        except ValueError:
            alpha = 0.01
        
        # Get training data source
        print("\nTraining data options:")
        print("1. Use sample corpus")
        print("2. Load from file")
        print("3. Enter text manually")
        
        data_choice = input("Choose data source (1-3): ").strip()
        
        try:
            self.model = NgramModel(n=n, smoothing_alpha=alpha)
            
            if data_choice == "1":
                # Use sample corpus
                corpus_path = os.path.join("data", "sample_corpus.txt")
                if os.path.exists(corpus_path):
                    print(f"Training on sample corpus: {corpus_path}")
                    self.model.train_from_file(corpus_path)
                else:
                    print("Error: Sample corpus not found")
                    return
            
            elif data_choice == "2":
                # Load from file
                file_path = input("Enter file path: ").strip()
                if os.path.exists(file_path):
                    print(f"Training on file: {file_path}")
                    self.model.train_from_file(file_path)
                else:
                    print("Error: File not found")
                    return
            
            elif data_choice == "3":
                # Manual text input
                print("Enter training text (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                
                if lines:
                    text = " ".join(lines)
                    print("Training on manual input...")
                    self.model.train_from_text(text)
                else:
                    print("No text entered")
                    return
            
            else:
                print("Invalid choice")
                return
            
            # Create predictor
            self.predictor = Predictor(self.model)
            print(f"\nModel trained successfully!")
            print(f"Vocabulary size: {self.model.vocab_size}")
            print(f"Total n-grams: {self.model.total_ngrams}")
            
            # Save model option
            save_choice = input("\nSave model to file? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = input("Enter filename (without extension): ").strip()
                if filename:
                    model_path = f"{filename}.pkl"
                    self.model.save_model(model_path)
                    print(f"Model saved to {model_path}")
        
        except Exception as e:
            print(f"Error training model: {str(e)}")
    
    def load_existing_model(self):
        """Load an existing trained model."""
        print("\n--- LOAD EXISTING MODEL ---")
        
        model_path = input("Enter model file path: ").strip()
        
        if not os.path.exists(model_path):
            print("Error: Model file not found")
            return
        
        try:
            self.model = NgramModel()
            self.model.load_model(model_path)
            self.predictor = Predictor(self.model)
            
            print(f"Model loaded successfully!")
            print(f"N-gram order: {self.model.n}")
            print(f"Vocabulary size: {self.model.vocab_size}")
            print(f"Total n-grams: {self.model.total_ngrams}")
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            self.predictor = None
    
    def interactive_prediction(self):
        """Start interactive prediction session."""
        if not self.model or not self.predictor:
            print("Error: No model loaded. Please train or load a model first.")
            return
        
        print("\n--- INTERACTIVE PREDICTION ---")
        print("Type text and get real-time predictions.")
        print("Commands: !exit (quit), !clear (clear session), !stats (show stats)")
        print("-" * 50)
        
        self.session = InteractivePredictionSession(self.predictor)
        
        while True:
            try:
                user_input = input("\nEnter text: ").strip()
                
                if user_input == "!exit":
                    break
                elif user_input == "!clear":
                    self.session.reset_session()
                    print("Session cleared")
                    continue
                elif user_input == "!stats":
                    self._show_session_stats()
                    continue
                elif not user_input:
                    continue
                
                # Add text to session
                self.session.add_text(user_input)
                
                # Get predictions
                predictions = self.session.predict_next(top_k=5)
                
                if predictions:
                    print(f"Predictions: {', '.join(predictions)}")
                    
                    # Option to accept a prediction
                    accept = input("Accept prediction? Enter number (1-5) or press Enter to continue: ").strip()
                    if accept.isdigit():
                        idx = int(accept) - 1
                        if 0 <= idx < len(predictions):
                            self.session.accept_prediction(predictions[idx])
                            print(f"Accepted: {predictions[idx]}")
                else:
                    print("No predictions available")
                
                # Show current session text
                session_text = self.session.get_session_text()
                if session_text:
                    print(f"Current text: {session_text}")
            
            except (EOFError, KeyboardInterrupt):
                break
        
        print("Interactive session ended")
    
    def batch_prediction(self):
        """Perform batch predictions on multiple inputs."""
        if not self.model or not self.predictor:
            print("Error: No model loaded. Please train or load a model first.")
            return
        
        print("\n--- BATCH PREDICTION ---")
        print("Enter multiple contexts for batch prediction.")
        print("Press Enter twice to finish input.")
        
        contexts = []
        print("Enter contexts (one per line):")
        while True:
            context = input().strip()
            if not context:
                break
            contexts.append(context)
        
        if not contexts:
            print("No contexts entered")
            return
        
        print("\n--- BATCH RESULTS ---")
        for i, context in enumerate(contexts, 1):
            predictions = self.predictor.predict(context, top_k=3)
            print(f"{i}. Context: '{context}'")
            print(f"   Predictions: {', '.join(predictions) if predictions else 'None'}")
    
    def model_evaluation(self):
        """Evaluate the current model."""
        if not self.model or not self.predictor:
            print("Error: No model loaded. Please train or load a model first.")
            return
        
        print("\n--- MODEL EVALUATION ---")
        
        # Load test data
        test_path = os.path.join("data", "test_data.txt")
        if not os.path.exists(test_path):
            print("Error: Test data file not found")
            return
        
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                test_sentences = [line.strip() for line in f if line.strip()]
            
            print("Running comprehensive evaluation...")
            evaluator = ModelEvaluator(self.model, self.predictor)
            results = evaluator.comprehensive_evaluation(test_sentences)
            
            # Display results
            report = evaluator.generate_evaluation_report(results)
            print(report)
            
            # Save report option
            save_choice = input("\nSave evaluation report? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = input("Enter filename (without extension): ").strip()
                if filename:
                    report_path = f"{filename}_evaluation.txt"
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"Report saved to {report_path}")
        
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
    
    def show_model_statistics(self):
        """Display current model statistics."""
        if not self.model:
            print("Error: No model loaded. Please train or load a model first.")
            return
        
        print("\n--- MODEL STATISTICS ---")
        stats = self.model.get_model_stats()
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Show top n-grams
        print("\nMost frequent n-grams:")
        sorted_ngrams = sorted(self.model.ngram_counts.items(), 
                             key=lambda x: x[1], reverse=True)
        
        for i, (ngram, count) in enumerate(sorted_ngrams[:10]):
            ngram_str = " ".join(ngram)
            print(f"{i+1:2d}. {ngram_str}: {count}")
    
    def _show_session_stats(self):
        """Show current session statistics."""
        if self.session:
            session_text = self.session.get_session_text()
            word_count = len(session_text.split()) if session_text else 0
            print(f"Session word count: {word_count}")
            print(f"Current context: {self.session.current_context}")
    
    def show_help(self):
        """Display help information."""
        print("\n--- HELP ---")
        print("Predictive Text Input System Help")
        print("-" * 40)
        print("1. Train new model: Create and train a new n-gram model")
        print("2. Load existing model: Load a previously saved model")
        print("3. Interactive prediction: Real-time text prediction session")
        print("4. Batch prediction: Test multiple contexts at once")
        print("5. Model evaluation: Comprehensive model performance analysis")
        print("6. Model statistics: View current model information")
        print()
        print("N-gram Orders:")
        print("- Bigram (n=2): Uses previous 1 word for prediction")
        print("- Trigram (n=3): Uses previous 2 words for prediction")
        print("- Higher orders: More context but require more data")
        print()
        print("Tips:")
        print("- Larger training corpora generally improve performance")
        print("- Higher n-gram orders need more training data")
        print("- Use smoothing to handle unseen word sequences")
    
    def run(self):
        """Run the interactive interface."""
        self.display_banner()
        
        while self.running:
            try:
                self.display_menu()
                choice = self.get_user_choice()
                
                if choice == "1":
                    self.train_new_model()
                elif choice == "2":
                    self.load_existing_model()
                elif choice == "3":
                    self.interactive_prediction()
                elif choice == "4":
                    self.batch_prediction()
                elif choice == "5":
                    self.model_evaluation()
                elif choice == "6":
                    self.show_model_statistics()
                elif choice == "7":
                    self.show_help()
                elif choice == "8":
                    print("\nThank you for using the Predictive Text System!")
                    self.running = False
                else:
                    print("Invalid choice. Please enter a number between 1 and 8.")
            
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Please try again.")


def main():
    """Main entry point for the interactive UI."""
    ui = InteractiveUI()
    ui.run()


if __name__ == "__main__":
    main()
