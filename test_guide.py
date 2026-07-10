"""
Quick Interactive Test Guide

Use this to test our model with words it knows!
"""

def show_test_guide():
    """Show words and phrases that work well with our model."""
    print("WORDS OUR MODEL KNOWS WELL")
    print("=" * 50)
    print("Try building sentences with these words:")
    print()
    
    print("ANIMAL PATTERNS:")
    print("   the + cat → predicts: sat")
    print("   cat + sat → predicts: on") 
    print("   sat + on → predicts: the")
    print("   on + the → predicts: mat")
    print()
    print("   the + dog → predicts: ran")
    print("   dog + ran → predicts: in")
    print("   ran + in → predicts: the") 
    print("   in + the → predicts: park")
    print()
    
    print("STORY PATTERNS:")
    print("   alice + fell → predicts: down")
    print("   fell + down → predicts: the")
    print("   down + the → predicts: rabbit")
    print("   the + rabbit → predicts: hole")
    print()
    
    print("DETECTIVE PATTERNS:")
    print("   sherlock + holmes → predicts: solved")
    print("   holmes + solved → predicts: the")
    print("   solved + the → predicts: mystery")
    print()
    print("   the + detective → predicts: examined")
    print("   detective + examined → predicts: the")
    print("   examined + the → predicts: evidence")
    print()
    
    print("FAIRY TALE PATTERNS:")
    print("   once + upon → predicts: a")
    print("   upon + a → predicts: time")
    print("   a + time → predicts: in")
    print()
    
    print("BOOK PATTERNS:")
    print("   elizabeth + read → predicts: a")
    print("   read + a → predicts: good")
    print("   a + good → predicts: book")
    print()
    
    print("TIPS FOR TESTING:")
    print("   - Start with 'the' - it's the most common word")
    print("   - Use simple, complete words (no contractions)")
    print("   - Try the patterns above for best results")
    print("   - If predictions are low, the model hasn't seen that pattern much")
    print()

if __name__ == "__main__":
    show_test_guide()
    
    print("Now run the interactive demo and try these patterns!")
    print("Command: python working_interactive_demo.py")
    print("Choose option 1 for interactive typing!")
