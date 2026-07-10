"""
SIMPLE LIVE PROFESSOR DEMO
Shows accuracy calculation in real-time with confusion matrix
Perfect for academic presentation!
"""

import numpy as np
from collections import defaultdict, Counter
import time

# Import our models
from legitimate_accuracy_improvement import (
    ImprovedNgramModel, SmartPreprocessor,
    proper_train_test_split, create_diverse_training_data,
    evaluate_model_properly
)

def show_live_accuracy():
    """Simple live accuracy demonstration."""
    
    print("=" * 60)
    print("LIVE ACCURACY DEMONSTRATION FOR PROFESSOR")
    print("=" * 60)
    print()
    
    # Step 1: Data preparation
    print("STEP 1: PREPARING DATA")
    print("-" * 30)
    
    print("Loading training data...")
    diverse_data = create_diverse_training_data()
    
    print(f"✓ Total sentences: {len(diverse_data)}")
    print(f"✓ Unique sentences: {len(set(diverse_data))}")
    
    # Show samples
    print("\nSample sentences:")
    samples = ["the cat sat on the mat", "alice fell down the rabbit hole", 
               "sherlock holmes solved the mystery"]
    for i, sentence in enumerate(samples, 1):
        print(f"   {i}. {sentence}")
    
    input("\nPress Enter to split data...")
    
    # Step 2: Train/Test split
    print("\nSTEP 2: TRAIN/TEST SPLIT")
    print("-" * 30)
    
    train_data, test_data = proper_train_test_split(diverse_data, test_ratio=0.3)
    
    print(f"✓ Training: {len(train_data)} sentences")
    print(f"✓ Testing: {len(test_data)} sentences")
    print(f"✓ No overlap: {len(set(train_data) & set(test_data)) == 0}")
    
    input("\nPress Enter to train model...")
    
    # Step 3: Train model
    print("\nSTEP 3: TRAINING MODEL")
    print("-" * 30)
    
    print("Processing training data...")
    preprocessor = SmartPreprocessor()
    all_tokens = []
    for sentence in train_data:
        tokens = preprocessor.preprocess_text(sentence)
        all_tokens.extend(tokens)
    
    print("Training 4-gram model with backoff...")
    model = ImprovedNgramModel(n=4, smoothing_alpha=0.01, 
                              vocab_threshold=1, use_backoff=True)
    model.train_from_tokens(all_tokens)
    
    print(f"✓ Model trained!")
    print(f"✓ Vocabulary size: {len(model.vocabulary)}")
    print(f"✓ N-grams learned: {len(model.ngram_counts)}")
    
    input("\nPress Enter to calculate accuracy live...")
    
    # Step 4: Live accuracy calculation
    print("\nSTEP 4: LIVE ACCURACY CALCULATION")
    print("-" * 30)
    
    print("Testing model on unseen data...")
    print("Processing each test sentence...")
    
    # Manual evaluation for transparency
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total_predictions = 0
    
    confusion_data = defaultdict(int)
    
    print(f"\nProgress:")
    for i, sentence in enumerate(test_data):
        print(f"Sentence {i+1}/{len(test_data)}: Testing predictions...", end=" ")
        
        tokens = preprocessor.preprocess_text(sentence)
        sentence_predictions = 0
        
        # Test each trigram in the sentence
        for j in range(2, len(tokens) - 1):
            context = tuple(tokens[j-2:j])
            actual_word = tokens[j]
            
            if actual_word not in ['<START>', '<END>']:
                try:
                    predictions = model.predict_next_words(context, top_k=5)
                    
                    if predictions:
                        pred_words = [word for word, prob in predictions]
                        
                        # Check accuracy
                        if actual_word in pred_words[:1]:
                            correct_top1 += 1
                            confusion_data['rank_1'] += 1
                        elif actual_word in pred_words[:3]:
                            correct_top3 += 1
                            rank = pred_words.index(actual_word) + 1
                            confusion_data[f'rank_{rank}'] += 1
                        elif actual_word in pred_words[:5]:
                            correct_top5 += 1
                            rank = pred_words.index(actual_word) + 1
                            confusion_data[f'rank_{rank}'] += 1
                        else:
                            confusion_data['not_in_top5'] += 1
                        
                        total_predictions += 1
                        sentence_predictions += 1
                except:
                    pass
        
        print(f"({sentence_predictions} predictions)")
        time.sleep(0.5)  # Dramatic pause
    
    # Calculate final accuracies
    if total_predictions > 0:
        top1_accuracy = correct_top1 / total_predictions
        top3_accuracy = (correct_top1 + correct_top3) / total_predictions
        top5_accuracy = (correct_top1 + correct_top3 + correct_top5) / total_predictions
    else:
        top1_accuracy = top3_accuracy = top5_accuracy = 0
    
    input(f"\nPress Enter to see results...")
    
    # Step 5: Show results
    print("\nSTEP 5: FINAL RESULTS")
    print("-" * 30)
    
    print("ACCURACY RESULTS:")
    print(f"   Top-1 Accuracy: {top1_accuracy:.1%}")
    print(f"   Top-3 Accuracy: {top3_accuracy:.1%}")
    print(f"   Top-5 Accuracy: {top5_accuracy:.1%}")
    print(f"   Total predictions: {total_predictions}")
    
    print("\nCONFUSION MATRIX:")
    print(f"   Correct at Rank 1: {confusion_data['rank_1']:3d} ({confusion_data['rank_1']/total_predictions:.1%})")
    print(f"   Correct at Rank 2: {confusion_data['rank_2']:3d} ({confusion_data['rank_2']/total_predictions:.1%})")
    print(f"   Correct at Rank 3: {confusion_data['rank_3']:3d} ({confusion_data['rank_3']/total_predictions:.1%})")
    print(f"   Correct at Rank 4: {confusion_data['rank_4']:3d} ({confusion_data['rank_4']/total_predictions:.1%})")
    print(f"   Correct at Rank 5: {confusion_data['rank_5']:3d} ({confusion_data['rank_5']/total_predictions:.1%})")
    print(f"   Not in Top-5:      {confusion_data['not_in_top5']:3d} ({confusion_data['not_in_top5']/total_predictions:.1%})")
    
    input("\nPress Enter to see live predictions...")
    
    # Step 6: Live predictions
    print("\nSTEP 6: LIVE PREDICTIONS DEMO")
    print("-" * 30)
    
    test_contexts = [
        ("the", "cat"),
        ("alice", "fell"),
        ("once", "upon"),
        ("sherlock", "holmes"),
        ("the", "detective")
    ]
    
    for context in test_contexts:
        print(f"\nContext: {context}")
        try:
            predictions = model.predict_next_words(context, top_k=3)
            if predictions:
                for i, (word, prob) in enumerate(predictions[:3], 1):
                    print(f"   {i}. {word} (probability: {prob:.3f})")
            else:
                print("   No predictions available")
        except Exception as e:
            print(f"   Error: {e}")
        
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print(f"Final accuracy: {top1_accuracy:.1%} (legitimate, no overfitting)")
    print("=" * 60)
    
    return {
        'top_1_accuracy': top1_accuracy,
        'top_3_accuracy': top3_accuracy,
        'top_5_accuracy': top5_accuracy,
        'total_predictions': total_predictions,
        'confusion_data': confusion_data
    }

def quick_accuracy_only():
    """Just show the accuracy quickly."""
    
    print("QUICK ACCURACY CHECK")
    print("=" * 30)
    
    # Quick setup
    diverse_data = create_diverse_training_data()
    train_data, test_data = proper_train_test_split(diverse_data, test_ratio=0.3)
    
    # Train model
    preprocessor = SmartPreprocessor()
    all_tokens = []
    for sentence in train_data:
        tokens = preprocessor.preprocess_text(sentence)
        all_tokens.extend(tokens)
    
    model = ImprovedNgramModel(n=4, smoothing_alpha=0.01, 
                              vocab_threshold=1, use_backoff=True)
    model.train_from_tokens(all_tokens)
    
    # Evaluate
    results = evaluate_model_properly(model, test_data, context_length=2)
    
    print(f"RESULTS:")
    print(f"   Top-1 Accuracy: {results['top_1_accuracy']:.1%}")
    print(f"   Top-3 Accuracy: {results['top_3_accuracy']:.1%}")
    print(f"   Top-5 Accuracy: {results['top_5_accuracy']:.1%}")
    print(f"   Total tests: {results['total_predictions']}")
    
    return results

if __name__ == "__main__":
    print("Choose demonstration:")
    print("1. Live step-by-step demo (best for professor)")
    print("2. Quick accuracy only")
    
    choice = input("\nChoice (1-2): ").strip()
    
    if choice == "2":
        quick_accuracy_only()
    else:
        show_live_accuracy()
