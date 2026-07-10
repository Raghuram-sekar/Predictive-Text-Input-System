"""
Text Preprocessing Module

This module provides comprehensive text preprocessing utilities for the
Predictive Text Input System. It handles cleaning, tokenization, and
normalization of text data for n-gram model training.
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


class TextPreprocessor:
    """
    A comprehensive text preprocessor for preparing text data for n-gram models.
    
    Features:
    - Lowercasing
    - Punctuation removal/handling
    - Tokenization
    - Sentence boundary detection
    - Optional stopword removal
    - Number handling
    """
    
    def __init__(self, 
                 remove_punctuation: bool = True,
                 remove_stopwords: bool = False,
                 handle_numbers: bool = True,
                 preserve_sentence_boundaries: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_punctuation: Whether to remove punctuation marks
            remove_stopwords: Whether to remove common stopwords
            handle_numbers: Whether to replace numbers with <NUM> token
            preserve_sentence_boundaries: Whether to add sentence boundary tokens
        """
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.handle_numbers = handle_numbers
        self.preserve_sentence_boundaries = preserve_sentence_boundaries
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize stopwords if needed
        if self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        
        # Special tokens
        self.sentence_start_token = '<START>'
        self.sentence_end_token = '<END>'
        self.number_token = '<NUM>'
        
    def _download_nltk_data(self):
        """Download required NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        if self.remove_stopwords:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Handle numbers
        if self.handle_numbers:
            text = re.sub(r'\b\d+\b', self.number_token, text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        tokens = word_tokenize(text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stopwords]
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline for text.
        
        Args:
            text: Raw input text
            
        Returns:
            List of processed tokens with optional sentence boundaries
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        if self.preserve_sentence_boundaries:
            # Process sentence by sentence
            sentences = self.tokenize_sentences(cleaned_text)
            all_tokens = []
            
            for sentence in sentences:
                # Tokenize sentence
                tokens = self.tokenize_words(sentence)
                
                if tokens:  # Only add if sentence has tokens
                    # Add sentence boundary tokens
                    sentence_tokens = [self.sentence_start_token] + tokens + [self.sentence_end_token]
                    all_tokens.extend(sentence_tokens)
            
            return all_tokens
        else:
            # Process entire text as one sequence
            return self.tokenize_words(cleaned_text)
    
    def preprocess_file(self, file_path: str, encoding: str = 'utf-8') -> List[str]:
        """
        Preprocess text from a file.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding
            
        Returns:
            List of processed tokens
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
            return self.preprocess_text(text)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {str(e)}")
    
    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """
        Preprocess a corpus of texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Combined list of processed tokens from all documents
        """
        all_tokens = []
        for text in texts:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
        return all_tokens
    
    def get_vocabulary(self, tokens: List[str]) -> set:
        """
        Extract vocabulary from tokenized text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Set of unique tokens (vocabulary)
        """
        return set(tokens)
    
    def get_statistics(self, tokens: List[str]) -> dict:
        """
        Get basic statistics about the processed text.
        
        Args:
            tokens: List of processed tokens
            
        Returns:
            Dictionary with text statistics
        """
        vocab = self.get_vocabulary(tokens)
        
        return {
            'total_tokens': len(tokens),
            'unique_tokens': len(vocab),
            'vocabulary_size': len(vocab),
            'average_token_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'longest_token': max(tokens, key=len) if tokens else '',
            'shortest_token': min(tokens, key=len) if tokens else ''
        }
