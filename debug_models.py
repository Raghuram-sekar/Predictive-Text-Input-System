"""
DEBUG VERSION - Let's see what's in the REAL trained models
"""

import pickle
import nltk
from nltk.tokenize import word_tokenize

def debug_models():
    """Debug the real trained models to see what's inside."""
    
    print("🔍 DEBUGGING REAL TRAINED MODELS")
    print("=" * 50)
    
    try:
        with open('real_trained_models.pkl', 'rb') as f:
            data = pickle.load(f)
            models = data['models']
            vocabularies = data['vocabularies']
        
        print(f"✅ Loaded {len(models)} models")
        
        # Check each model
        for model_name, model in models.items():
            vocab = vocabularies[model_name]
            print(f"\n📊 MODEL: {model_name}")
            print(f"  • Type: {type(model).__name__}")
            print(f"  • N-gram order: {model.order}")
            print(f"  • Vocabulary size: {len(vocab):,}")
            
            # Sample vocabulary
            sample_vocab = list(vocab)[:20]
            print(f"  • Sample vocab: {sample_vocab}")
            
            # Test simple predictions
            print(f"  • Testing predictions...")
            
            # Try different contexts
            test_contexts = [
                ['i'],
                ['the'],
                ['and'],
                ['<s>'],
                ['<s>', 'i'],
                ['<s>', 'the']
            ]
            
            for context in test_contexts:
                try:
                    # Test with a common word
                    test_word = 'am'
                    if test_word in vocab:
                        score = model.score(test_word, context)
                        print(f"    Context {context} + '{test_word}' = {score:.10f}")
                    
                    # Try generating predictions
                    candidates = ['am', 'is', 'was', 'have', 'had', 'will', 'can', 'do']
                    predictions = []
                    
                    for word in candidates:
                        if word in vocab:
                            try:
                                score = model.score(word, context)
                                if score > 1e-20:
                                    predictions.append((word, score))
                            except:
                                pass
                    
                    if predictions:
                        predictions.sort(key=lambda x: x[1], reverse=True)
                        print(f"    Best for {context}: {predictions[:3]}")
                        break
                
                except Exception as e:
                    print(f"    Error with {context}: {e}")
            
            print()
    
    except Exception as e:
        print(f"❌ Error loading models: {e}")

if __name__ == "__main__":
    debug_models()