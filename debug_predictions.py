"""
DEBUG: Check if the mathematical calculations are really working
"""

import pickle
import nltk
from nltk.tokenize import word_tokenize

def debug_predictions():
    """Debug the actual predictions and probabilities."""
    
    print("🔍 DEBUGGING MATHEMATICAL CALCULATIONS")
    print("=" * 60)
    
    try:
        with open('real_trained_models.pkl', 'rb') as f:
            data = pickle.load(f)
            models = data['models']
            vocabularies = data['vocabularies']
        
        print(f"✅ Loaded {len(models)} models")
        
        # Test with "coming over to"
        test_phrase = "coming over to"
        print(f"\n🧪 TESTING: '{test_phrase}'")
        print("=" * 40)
        
        for model_name, model in models.items():
            if 'Bigram' in model_name:  # Test bigram models
                print(f"\n📊 MODEL: {model_name}")
                vocab = vocabularies[model_name]
                
                # Prepare context
                tokens = word_tokenize(test_phrase.lower())
                context = tuple(tokens[-1:])  # Just last word for bigram
                print(f"Context: {context}")
                
                # Test specific words that should make sense
                test_words = ['your', 'my', 'the', 'be', 'see', 'visit', 'help']
                
                print("Word probabilities:")
                for word in test_words:
                    if word in vocab:
                        try:
                            prob = model.score(word, context)
                            print(f"  {word:<8} → {prob:.8f}")
                        except Exception as e:
                            print(f"  {word:<8} → ERROR: {e}")
                    else:
                        print(f"  {word:<8} → NOT IN VOCAB")
                
                # Get top 10 predictions manually
                print("\nTop 10 actual predictions:")
                predictions = []
                candidate_words = list(vocab)[:2000]  # Sample vocabulary
                
                for word in candidate_words:
                    if word not in ['<s>', '</s>'] and word.isalpha() and len(word) > 1:
                        try:
                            prob = model.score(word, context)
                            if prob > 1e-10:
                                predictions.append((word, prob))
                        except:
                            continue
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                for i, (word, prob) in enumerate(predictions[:10], 1):
                    print(f"  {i:2d}. {word:<12} → {prob:.8f}")
                
                break  # Just test first bigram model
    
    except Exception as e:
        print(f"❌ Error: {e}")

def test_simple_predictions():
    """Test very simple predictions to see if math works."""
    print("\n🧮 TESTING SIMPLE MATH")
    print("=" * 30)
    
    try:
        with open('real_trained_models.pkl', 'rb') as f:
            data = pickle.load(f)
            models = data['models']
            vocabularies = data['vocabularies']
        
        # Get a simple model
        model_name = 'Bigram MLE'
        model = models[model_name]
        vocab = vocabularies[model_name]
        
        print(f"Testing {model_name}")
        print(f"Vocabulary size: {len(vocab):,}")
        
        # Test very common words
        common_contexts = ['i', 'you', 'the', 'to']
        
        for context_word in common_contexts:
            if context_word in vocab:
                print(f"\nContext: '{context_word}'")
                
                # Get probabilities for common next words
                next_words = ['am', 'are', 'was', 'will', 'have', 'can', 'do']
                total_prob = 0
                
                for next_word in next_words:
                    if next_word in vocab:
                        try:
                            prob = model.score(next_word, tuple([context_word]))
                            total_prob += prob
                            print(f"  {context_word} {next_word} → {prob:.6f}")
                        except Exception as e:
                            print(f"  {context_word} {next_word} → ERROR: {e}")
                
                print(f"  Total probability for tested words: {total_prob:.6f}")
                
                # Check if probabilities sum correctly
                if total_prob > 0:
                    print("  ✅ Math seems to be working")
                else:
                    print("  ❌ No probabilities found - math might be broken")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def check_training_data_quality():
    """Check if the training data actually contains sensible phrases."""
    print("\n📚 CHECKING TRAINING DATA QUALITY")
    print("=" * 40)
    
    corpus_file = "datasets/modern/modern_conversational_corpus.txt"
    
    try:
        with open(corpus_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        print(f"Corpus size: {len(content):,} characters")
        
        # Search for our test phrase
        test_phrases = [
            "coming over to",
            "my name is",
            "how are you",
            "i am going",
            "what are you"
        ]
        
        for phrase in test_phrases:
            count = content.lower().count(phrase.lower())
            print(f"'{phrase}' appears {count} times")
            
            if count > 0:
                # Find some examples
                lower_content = content.lower()
                start = lower_content.find(phrase.lower())
                if start >= 0:
                    # Get context around the phrase
                    context_start = max(0, start - 50)
                    context_end = min(len(content), start + len(phrase) + 50)
                    context = content[context_start:context_end].replace('\n', ' ')
                    print(f"  Example: ...{context}...")
        
    except Exception as e:
        print(f"❌ Error reading corpus: {e}")

if __name__ == "__main__":
    debug_predictions()
    test_simple_predictions()
    check_training_data_quality()
