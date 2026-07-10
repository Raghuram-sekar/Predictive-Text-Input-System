"""
🔍 TEST ALL MODELS - Check if each model gives different predictions
"""

import pickle
import nltk
from nltk.tokenize import word_tokenize

def test_all_models():
    """Test all 8 models to verify they give different predictions."""
    
    print("🔍 TESTING ALL 8 MODELS FOR DIFFERENT PREDICTIONS")
    print("=" * 70)
    
    try:
        with open('real_trained_models.pkl', 'rb') as f:
            data = pickle.load(f)
            models = data['models']
            vocabularies = data['vocabularies']
        
        print(f"✅ Loaded {len(models)} models")
        
        # Test phrases
        test_phrases = [
            "i am",
            "i love", 
            "my name",
            "how are",
            "coming over to"
        ]
        
        for phrase in test_phrases:
            print(f"\n🧪 TESTING PHRASE: '{phrase}'")
            print("=" * 50)
            
            # Test each model
            for model_name, model in models.items():
                vocab = vocabularies[model_name]
                
                # Prepare context
                tokens = word_tokenize(phrase.lower())
                n = model.order
                
                if n == 2:  # Bigram
                    context = tuple(tokens[-1:])
                else:  # Trigram
                    if len(tokens) >= 2:
                        context = tuple(tokens[-2:])
                    else:
                        context = tuple(['<s>'] + tokens)
                
                print(f"\n📊 {model_name} (order={n}, context={context}):")
                
                # Get top 5 predictions
                predictions = []
                candidates = list(vocab)[:1000]  # Sample for speed
                
                for word in candidates:
                    if word not in ['<s>', '</s>'] and word.isalpha() and len(word) > 1:
                        try:
                            prob = model.score(word, context)
                            if prob > 1e-12:
                                predictions.append((word, prob))
                        except:
                            continue
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Show top 5
                for i, (word, prob) in enumerate(predictions[:5], 1):
                    print(f"  {i}. {word:<12} → {prob:.8f}")
                
                if not predictions:
                    print("  No predictions found")
        
        # Summary comparison
        print(f"\n📈 SUMMARY: COMPARING DIFFERENT MODEL TYPES")
        print("=" * 60)
        
        test_phrase = "i am"
        tokens = word_tokenize(test_phrase.lower())
        
        print(f"Testing '{test_phrase}' across all models:")
        print()
        
        model_results = {}
        
        for model_name, model in models.items():
            vocab = vocabularies[model_name]
            n = model.order
            
            if n == 2:
                context = tuple(tokens[-1:])
            else:
                context = tuple(['<s>'] + tokens) if len(tokens) < 2 else tuple(tokens[-2:])
            
            # Get top 3 predictions
            predictions = []
            candidates = list(vocab)[:1000]
            
            for word in candidates:
                if word not in ['<s>', '</s>'] and word.isalpha() and len(word) > 1:
                    try:
                        prob = model.score(word, context)
                        if prob > 1e-12:
                            predictions.append((word, prob))
                    except:
                        continue
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_words = [word for word, prob in predictions[:3]]
            model_results[model_name] = top_words
            
            print(f"{model_name:<25} → {top_words}")
        
        # Check if models are truly different
        print(f"\n🔍 UNIQUENESS CHECK:")
        print("-" * 30)
        
        unique_predictions = set()
        for model_name, predictions in model_results.items():
            pred_tuple = tuple(predictions)
            unique_predictions.add(pred_tuple)
            
        print(f"Total models: {len(model_results)}")
        print(f"Unique prediction sets: {len(unique_predictions)}")
        
        if len(unique_predictions) > 1:
            print("✅ MODELS ARE GENUINELY DIFFERENT!")
            print("   Different algorithms produce different predictions.")
        else:
            print("⚠️  All models giving same predictions - might be an issue")
        
        # Test specific algorithm differences
        print(f"\n🔬 ALGORITHM-SPECIFIC TESTS:")
        print("-" * 40)
        
        # Compare MLE vs Laplace (should handle unseen n-grams differently)
        mle_model = models.get('Bigram MLE')
        laplace_model = models.get('Bigram Laplace')
        
        if mle_model and laplace_model:
            test_word = "going"
            context = ('am',)
            
            mle_score = mle_model.score(test_word, context)
            laplace_score = laplace_model.score(test_word, context)
            
            print(f"MLE score for 'am going':      {mle_score:.8f}")
            print(f"Laplace score for 'am going': {laplace_score:.8f}")
            
            if abs(mle_score - laplace_score) > 1e-8:
                print("✅ MLE and Laplace give different scores (correct!)")
            else:
                print("⚠️  MLE and Laplace giving same scores")
        
        # Compare Bigram vs Trigram (should use different context)
        bigram_model = models.get('Bigram MLE')
        trigram_model = models.get('Trigram MLE')
        
        if bigram_model and trigram_model:
            test_phrase = "i am"
            tokens = word_tokenize(test_phrase.lower())
            
            bigram_context = tuple(tokens[-1:])  # ('am',)
            trigram_context = tuple(['<s>'] + tokens)  # ('<s>', 'i', 'am')
            
            test_word = "going"
            
            bigram_score = bigram_model.score(test_word, bigram_context)
            trigram_score = trigram_model.score(test_word, trigram_context)
            
            print(f"Bigram context: {bigram_context} → {bigram_score:.8f}")
            print(f"Trigram context: {trigram_context} → {trigram_score:.8f}")
            
            if abs(bigram_score - trigram_score) > 1e-8:
                print("✅ Bigram and Trigram use different contexts (correct!)")
            else:
                print("⚠️  Bigram and Trigram giving same scores")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_all_models()