"""
Smart Context Handling for N-gram Models

This module implements:
1. Variable-length context windows
2. Skip-gram patterns
3. Better sentence boundary handling
4. Context clustering
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
import re


class SmartContextHandler:
    """
    Implements advanced context handling techniques for better predictions.
    Memory-efficient implementation using sparse matrices and batch processing.
    """
    
    def __init__(self, max_window: int = 5, skip_size: int = 2, max_contexts: int = 50000):
        """
        Initialize smart context handler.
        
        Args:
            max_window: Maximum context window size
            skip_size: Maximum number of words to skip in skip-grams
            max_contexts: Maximum number of contexts to process at once (larger means faster but more memory)
        """
        self.max_window = max_window
        self.skip_size = skip_size
        self.max_contexts = max_contexts
        self.context_vectors = {}
        self.context_clusters = None
        self.cluster_model = None
        
        # For progress tracking
        self.total_processed = 0
        
        # Special tokens for sentence boundaries
        self.start_token = "<START>"
        self.end_token = "<END>"
    
    def get_variable_contexts(self, tokens: List[str], target_idx: int) -> List[Tuple[str, ...]]:
        """
        Get multiple context windows of varying lengths.
        
        Args:
            tokens: List of tokens
            target_idx: Index of target word
            
        Returns:
            List of context tuples of varying lengths
        """
        contexts = []
        
        # Get contexts of different lengths
        for window in range(1, self.max_window + 1):
            start_idx = max(0, target_idx - window)
            context = tuple(tokens[start_idx:target_idx])
            if context:
                contexts.append(context)
        
        return contexts
    
    def get_skip_gram_contexts(self, tokens: List[str], target_idx: int) -> List[Tuple[str, ...]]:
        """
        Get skip-gram patterns from context.
        
        Args:
            tokens: List of tokens
            target_idx: Index of target word
            
        Returns:
            List of skip-gram context tuples
        """
        skip_contexts = []
        
        # Generate skip-gram patterns
        for window in range(2, self.max_window + 1):
            for skip in range(1, min(self.skip_size + 1, window)):
                start_idx = max(0, target_idx - window)
                context = tokens[start_idx:target_idx]
                
                # Generate skip patterns
                for i in range(len(context) - skip):
                    skip_context = context[:i] + context[i + skip:]
                    if skip_context:
                        skip_contexts.append(tuple(skip_context))
        
        return skip_contexts
    
    def handle_sentence_boundaries(self, text: str) -> List[str]:
        """
        Add explicit sentence boundary tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens with boundary markers
        """
        # Split into sentences (basic implementation)
        sentences = re.split(r'[.!?]+', text)
        
        tokens = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Add boundary tokens
                sentence_tokens = sentence.split()
                if sentence_tokens:
                    tokens.extend([self.start_token] + sentence_tokens + [self.end_token])
        
        return tokens
    
    def build_context_vectors(self, contexts: List[Tuple[str, ...]], vocabulary: Set[str]):
        """
        Build vector representations for contexts using sparse matrices and optimized processing.
        Processes all contexts at once for maximum efficiency.
        
        Args:
            contexts: List of context tuples
            vocabulary: Set of known words
        """
        print("Building context vectors with optimized processing...")
        print(f"Total contexts to process: {len(contexts):,}")
        print(f"Vocabulary size: {len(vocabulary):,}")
        
        # Create vocabulary index
        vocab_index = {word: i for i, word in enumerate(vocabulary)}
        vector_size = len(vocabulary)
        
        # Process all contexts efficiently at once
        print("Building sparse matrix...")
        # Initialize data structures for sparse matrix
        rows = []
        cols = []
        data = []
        
        # Process all contexts with progress updates
        for ctx_idx, context in enumerate(contexts):
            # Build word counts for this context
            word_counts = defaultdict(int)
            for word in context:
                if word in vocab_index:
                    word_counts[vocab_index[word]] += 1
            
            # Add entries for this context
            for word_idx, count in word_counts.items():
                rows.append(ctx_idx)
                cols.append(word_idx)
                data.append(count)
            
            # Show progress every 100K contexts
            if (ctx_idx + 1) % 100000 == 0 or ctx_idx == len(contexts) - 1:
                print(f"\rProcessed {ctx_idx + 1:,}/{len(contexts):,} contexts...", end='')
        
        print("\nCreating final sparse matrix...")
        # Create single sparse matrix for all contexts
        from scipy.sparse import csr_matrix
        vectors = csr_matrix((data, (rows, cols)), shape=(len(contexts), vector_size))
        
        # Normalize the vectors using memory-efficient chunking
        print("Normalizing vectors...")
        # Calculate row sums without converting to dense
        row_sums = np.asarray(vectors.sum(axis=1)).ravel()
        
        # Process normalization in chunks to avoid memory issues
        chunk_size = 1000  # Adjust based on available memory
        n_chunks = (len(contexts) + chunk_size - 1) // chunk_size
        
        for chunk in range(n_chunks):
            start_idx = chunk * chunk_size
            end_idx = min((chunk + 1) * chunk_size, len(contexts))
            
            # Get chunk of vectors and ensure it's CSR format
            chunk_vectors = vectors[start_idx:end_idx].tocsr()
            chunk_sums = row_sums[start_idx:end_idx]
            chunk_nonzero = chunk_sums > 0
            
            # Normalize chunk
            if np.any(chunk_nonzero):
                chunk_vectors = chunk_vectors.multiply(1/chunk_sums[:, np.newaxis]).tocsr()
            
            # Store normalized vectors row by row
            for i, context in enumerate(contexts[start_idx:end_idx]):
                self.context_vectors[context] = chunk_vectors.getrow(i)
            
            if (chunk + 1) % 10 == 0 or chunk == n_chunks - 1:
                print(f"\rNormalized {end_idx:,}/{len(contexts):,} vectors...", end="")
            
            if (chunk + 1) % 10 == 0 or chunk == n_chunks - 1:
                print(f"\rNormalized {end_idx}/{len(contexts)} vectors...", end="")
        
        print("Context vectors built and normalized successfully!")
        
        return vectors
    
    def cluster_contexts(self, n_clusters: int = 10):
        """
        Cluster similar contexts together using sparse matrix operations.
        
        Args:
            n_clusters: Number of context clusters
        """
        if not self.context_vectors:
            raise ValueError("Context vectors not built. Call build_context_vectors first.")
        
        print("\nPreparing vectors for clustering...")
        # Prepare data for clustering
        contexts = list(self.context_vectors.keys())
        print(f"Found {len(contexts):,} contexts to cluster")
        
        # Stack sparse vectors directly without converting to dense
        from scipy.sparse import vstack
        vectors_list = []
        
        for context in contexts:
            vec = self.context_vectors[context]
            if hasattr(vec, 'toarray'):  # If it's a sparse matrix
                vectors_list.append(vec)
            else:
                # Convert to sparse if somehow it's dense
                from scipy.sparse import csr_matrix
                vectors_list.append(csr_matrix(vec))
        
        # Stack all sparse vectors
        vectors = vstack(vectors_list)
        print(f"Created sparse matrix of shape {vectors.shape}")
        
        # Use MiniBatchKMeans for large datasets with sparse matrices
        from sklearn.cluster import MiniBatchKMeans
        print(f"Clustering {vectors.shape[0]:,} contexts into {n_clusters} clusters using MiniBatch...")
        
        # MiniBatchKMeans works better with sparse matrices and large datasets
        self.cluster_model = MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=1000,
            n_init=3,
            random_state=42
        )
        clusters = self.cluster_model.fit_predict(vectors)
        
        print("Clustering completed successfully")
        
        # Store cluster assignments
        self.context_clusters = defaultdict(list)
        for context, cluster in zip(contexts, clusters):
            self.context_clusters[cluster].append(context)
        
        # Print cluster distribution
        cluster_sizes = [len(self.context_clusters[i]) for i in range(n_clusters)]
        print(f"Cluster sizes: {cluster_sizes}")
        print(f"Average cluster size: {np.mean(cluster_sizes):.1f}")
        print(f"Largest cluster: {max(cluster_sizes)}, Smallest cluster: {min(cluster_sizes)}")
    
    def get_similar_contexts(self, context: Tuple[str, ...], n: int = 5) -> List[Tuple[str, ...]]:
        """
        Get similar contexts based on clustering.
        
        Args:
            context: Input context
            n: Number of similar contexts to return
            
        Returns:
            List of similar context tuples
        """
        if context not in self.context_vectors:
            return []
        
        # Get context's cluster
        vector = self.context_vectors[context]
        if hasattr(vector, 'toarray'):  # If it's a sparse matrix
            vector = vector.toarray().flatten()
        cluster = self.cluster_model.predict([vector])[0]
        
        # Get contexts from same cluster
        similar = self.context_clusters[cluster]
        
        # Sort by similarity to input context
        similarities = []
        for other in similar:
            if other != context:
                sim = np.dot(vector, self.context_vectors[other])
                similarities.append((sim, other))
        
        # Return top-n similar contexts
        return [c for _, c in sorted(similarities, reverse=True)[:n]]


class EnhancedContextPredictor:
    """
    Enhanced prediction using smart context handling.
    """
    
    def __init__(self, base_model, context_handler: SmartContextHandler):
        """
        Initialize enhanced predictor.
        
        Args:
            base_model: Base n-gram model
            context_handler: Smart context handler
        """
        self.base_model = base_model
        self.context_handler = context_handler
    
    def predict_next_word(self, context: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next word using enhanced context handling.
        
        Args:
            context: Input context
            k: Number of predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        # Get various context patterns
        var_contexts = self.context_handler.get_variable_contexts(context, len(context))
        skip_contexts = self.context_handler.get_skip_gram_contexts(context, len(context))
        
        # Get predictions for each context
        predictions = defaultdict(float)
        
        # Variable-length contexts
        for ctx in var_contexts:
            ctx_preds = self.base_model.predict(ctx)
            weight = len(ctx) / self.context_handler.max_window
            for word, prob in ctx_preds:
                predictions[word] += prob * weight
        
        # Skip-gram contexts
        for ctx in skip_contexts:
            ctx_preds = self.base_model.predict(ctx)
            weight = 0.5  # Lower weight for skip-grams
            for word, prob in ctx_preds:
                predictions[word] += prob * weight
        
        # Normalize predictions
        total = sum(predictions.values())
        if total > 0:
            predictions = {w: p/total for w, p in predictions.items()}
        
        # Return top-k predictions
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:k]