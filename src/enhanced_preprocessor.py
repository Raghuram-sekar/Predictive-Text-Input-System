"""
Enhanced Text Preprocessing Pipeline

This module implements an advanced preprocessing pipeline with:
1. Smart contraction handling
2. Intelligent number normalization
3. Advanced tokenization
4. Context-aware case handling
5. Special token preservation
"""

import re
from typing import List, Dict, Set, Tuple
import unicodedata
import inflect
from collections import defaultdict


class EnhancedPreprocessor:
    """
    Advanced text preprocessing for better n-gram model performance.
    Handles contractions, numbers, case, and special tokens intelligently.
    """
    
    def __init__(self):
        """Initialize the enhanced preprocessor."""
        self.inflect_engine = inflect.engine()
        
        # Common English contractions
        self.contractions = {
            "ain't": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have"
        }
        
        # Special tokens to preserve
        self.special_tokens = {
            "<START>", "<END>", "<UNK>", "<NUM>", "<URL>", "<EMAIL>",
            "<PERSON>", "<ORG>", "<DATE>", "<TIME>", "<MONEY>"
        }
        
        # Common abbreviations
        self.abbreviations = {
            "mr.": "mister",
            "mrs.": "missus",
            "dr.": "doctor",
            "prof.": "professor",
            "sr.": "senior",
            "jr.": "junior",
            "vs.": "versus",
            "etc.": "etcetera",
            "approx.": "approximately",
            "apt.": "apartment",
            "dept.": "department",
            "est.": "established",
            "feb.": "february",
            "jan.": "january",
            "mar.": "march",
            "apr.": "april",
            "aug.": "august",
            "sep.": "september",
            "oct.": "october",
            "nov.": "november",
            "dec.": "december"
        }
    
    def preprocess_text(self, text: str, preserve_case: bool = False) -> List[str]:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            
        Returns:
            List of preprocessed tokens
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Handle URLs and emails
        text = self._replace_urls_and_emails(text)
        
        # Handle contractions
        text = self._expand_contractions(text)
        
        # Handle numbers
        text = self._normalize_numbers(text)
        
        # Handle abbreviations
        text = self._expand_abbreviations(text)
        
        # Tokenize
        tokens = self._advanced_tokenize(text)
        
        # Handle case
        if not preserve_case:
            tokens = [t.lower() for t in tokens]
        
        return tokens
    
    def _replace_urls_and_emails(self, text: str) -> str:
        """Replace URLs and email addresses with special tokens."""
        # URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '<URL>', text)
        
        # Email pattern
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        text = re.sub(email_pattern, '<EMAIL>', text)
        
        return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions to their full forms."""
        # Sort contractions by length (longest first)
        contractions = sorted(self.contractions.items(), key=lambda x: len(x[0]), reverse=True)
        
        for contraction, expansion in contractions:
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Convert numbers to a normalized format."""
        def replace_number(match):
            num = match.group()
            try:
                # Convert number words
                if num.isdigit():
                    word = self.inflect_engine.number_to_words(num)
                    return word.replace('-', ' ')
                return num
            except:
                return '<NUM>'
        
        # Replace numbers with words
        text = re.sub(r'\b\d+\b', replace_number, text)
        
        # Handle special number cases
        text = re.sub(r'\d+(?:\.\d+)?%', lambda x: self.inflect_engine.number_to_words(int(float(x.group()[:-1]))) + ' percent', text)
        text = re.sub(r'\$\d+(?:\.\d+)?', lambda x: self.inflect_engine.number_to_words(int(float(x.group()[1:]))) + ' dollars', text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _advanced_tokenize(self, text: str) -> List[str]:
        """
        Advanced tokenization with special token handling.
        
        Features:
        1. Preserves special tokens
        2. Handles hyphenated words
        3. Preserves sentence boundaries
        4. Handles punctuation intelligently
        """
        # Preserve special tokens
        preserved_tokens = {}
        for i, token in enumerate(self.special_tokens):
            if token in text:
                placeholder = f"PLACEHOLDER_{i}"
                preserved_tokens[placeholder] = token
                text = text.replace(token, placeholder)
        
        # Split on whitespace and punctuation
        tokens = []
        for word in text.split():
            # Handle hyphenated words
            if '-' in word and not any(c.isdigit() for c in word):
                parts = word.split('-')
                if all(len(p) > 1 for p in parts):  # Only split if parts are substantial
                    tokens.extend(parts)
                    continue
            
            # Handle punctuation
            puncts = []
            while word and word[-1] in '.!?,:;)]}':
                puncts.insert(0, word[-1])
                word = word[:-1]
            while word and word[0] in '([{':
                puncts.append(word[0])
                word = word[1:]
            
            if word:
                tokens.append(word)
            if puncts:
                tokens.extend(puncts)
        
        # Restore special tokens
        tokens = [preserved_tokens.get(t, t) for t in tokens]
        
        return tokens
    
    def mark_entities(self, text: str) -> str:
        """
        Mark named entities with special tokens.
        Requires additional NLP tools for full functionality.
        """
        # This is a placeholder for named entity recognition
        # In practice, you would use a proper NER tool
        return text


def preprocess_file(filepath: str, preprocessor: EnhancedPreprocessor = None) -> List[str]:
    """
    Preprocess an entire file.
    
    Args:
        filepath: Path to text file
        preprocessor: Optional custom preprocessor
        
    Returns:
        List of preprocessed tokens
    """
    if preprocessor is None:
        preprocessor = EnhancedPreprocessor()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return preprocessor.preprocess_text(text)


if __name__ == "__main__":
    # Example usage
    preprocessor = EnhancedPreprocessor()
    
    text = "Mr. Smith's can't believe it's $123.45! Check http://example.com or email@test.com"
    tokens = preprocessor.preprocess_text(text)
    print("Original:", text)
    print("Preprocessed:", " ".join(tokens))