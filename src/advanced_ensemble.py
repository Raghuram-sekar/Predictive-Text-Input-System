"""
Advanced Ensemble Techniques for N-gram Models

This module implements:
1. Weighted voting between n-gram orders
2. Context-specific model selection
3. Domain-specific model combinations
4. Adaptive confidence-based weighting
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression


class AdvancedEnsemble:
    """
    Advanced ensemble combining multiple n-gram models with smart weighting.
    """
    
    def __init__(self, models: List[object], domains: Optional[List[str]] = None):
        """
        Initialize advanced ensemble.
        
        Args:
            models: List of n-gram models
            domains: Optional list of domain names for domain-specific models
        """
        self.models = models
        self.domains = domains
        
        # Confidence scoring model
        self.confidence_model = LogisticRegression()
        
        # Context-specific weights
        self.context_weights = {}
        
        # Domain-specific weights
        self.domain_weights = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.model_performance = defaultdict(list)
    
    def train_ensemble(self, validation_data: List[Tuple[List[str], str]]):
        """
        Train ensemble weights using validation data.
        
        Args:
            validation_data: List of (context, target_word) pairs
        """
        print("Training ensemble weights...")
        
        if not validation_data:
            print("No validation data provided, using equal weights")
            # Set equal weights for all models
            self.base_weights = {model: 1.0/len(self.models) for model in self.models}
            return
        
        # Collect predictions from all models
        X = []  # Features for confidence model
        y = []  # Binary success labels
        model_accuracies = defaultdict(list)
        
        for context, target in validation_data[:min(100, len(validation_data))]:  # Limit for speed
            # Get predictions from each model
            for model in self.models:
                try:
                    predictions = model.predict(context, top_k=5)
                    if predictions:
                        top_pred = predictions[0][0]
                        prob = predictions[0][1] 
                        
                        # Track performance
                        success = (top_pred == target)
                        model_accuracies[model].append(float(success))
                        
                        # Features: probability, context length, etc.
                        X.append([prob, len(context), float(success)])
                        y.append(1 if success else 0)
                    else:
                        model_accuracies[model].append(0.0)
                        X.append([0.0, len(context), 0.0])
                        y.append(0)
                except Exception as e:
                    print(f"Error getting predictions from model: {e}")
                    model_accuracies[model].append(0.0)
        
        # Calculate base weights from performance
        self.base_weights = {}
        for model in self.models:
            if model in model_accuracies and model_accuracies[model]:
                accuracy = sum(model_accuracies[model]) / len(model_accuracies[model])
                self.base_weights[model] = accuracy
            else:
                self.base_weights[model] = 0.1  # Small default weight
        
        # Normalize weights
        total = sum(self.base_weights.values())
        if total > 0:
            self.base_weights = {m: w/total for m, w in self.base_weights.items()}
        else:
            # Fallback to equal weights
            self.base_weights = {model: 1.0/len(self.models) for model in self.models}
        
        print("Ensemble training complete!")
    
    def predict_with_confidence(self, context: List[str], domain: str = "general") -> List[Tuple[str, float]]:
        """
        Make ensemble predictions with confidence scoring.
        
        Args:
            context: Input context
            domain: Domain identifier (unused in this implementation)
            
        Returns:
            List of (word, probability) tuples
        """
        if not self.models:
            return []
        
        # Get predictions from all models
        all_predictions = defaultdict(float)
        
        for model in self.models:
            try:
                predictions = model.predict(context, top_k=10)
                weight = self.base_weights.get(model, 1.0/len(self.models))
                
                for word, prob in predictions:
                    all_predictions[word] += prob * weight
            except Exception as e:
                print(f"Error getting predictions from model: {e}")
                continue
        
        # Sort by combined probability
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:5]  # Return top 5
    
    def train_domain_weights(self, domain_data: Dict[str, List[Tuple[List[str], str]]]):
        """
        Train domain-specific weights.
        
        Args:
            domain_data: Dictionary mapping domains to validation data
        """
        print("Training domain-specific weights...")
        
        for domain, data in domain_data.items():
            domain_success = defaultdict(list)
            
            # Evaluate each model on domain data
            for context, target in data:
                for model in self.models:
                    predictions = model.predict(context)
                    if predictions:
                        success = (predictions[0][0] == target)
                        domain_success[model].append(success)
            
            # Calculate domain-specific weights
            for model, successes in domain_success.items():
                if successes:
                    accuracy = sum(successes) / len(successes)
                    self.domain_weights[domain][model] = accuracy
            
            # Normalize weights for domain
            total = sum(self.domain_weights[domain].values())
            if total > 0:
                for model in self.domain_weights[domain]:
                    self.domain_weights[domain][model] /= total
        
        print("Domain weight training complete!")
    
    def get_context_features(self, context: List[str]) -> List[float]:
        """
        Extract features from context for confidence model.
        
        Args:
            context: Input context
            
        Returns:
            List of context features
        """
        features = []
        
        for model in self.models:
            predictions = model.predict(context)
            prob = predictions[0][1] if predictions else 0
            
            features.extend([
                prob,
                len(context),
                self.base_weights[model]
            ])
        
        return features
    
    def predict_with_confidence(self, context: List[str], domain: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Make predictions using confidence-weighted ensemble.
        
        Args:
            context: Input context
            domain: Optional domain identifier
            
        Returns:
            List of (word, probability) tuples
        """
        predictions = defaultdict(float)
        
        # Get confidence score
        features = self.get_context_features(context)
        confidence = self.confidence_model.predict_proba([features])[0][1]
        
        # Get predictions from each model
        for model in self.models:
            # Get model weight
            base_weight = self.base_weights[model]
            domain_weight = self.domain_weights[domain][model] if domain else base_weight
            
            # Adjust weight by confidence
            weight = base_weight * 0.3 + domain_weight * 0.3 + confidence * 0.4
            
            # Add weighted predictions
            model_preds = model.predict(context)
            for word, prob in model_preds:
                predictions[word] += prob * weight
        
        # Normalize predictions
        total = sum(predictions.values())
        if total > 0:
            predictions = {w: p/total for w, p in predictions.items()}
        
        # Return sorted predictions
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)


class DomainAwareEnsemble:
    """
    Ensemble that specializes in different text domains.
    """
    
    def __init__(self, domain_models: Dict[str, List[object]]):
        """
        Initialize domain-aware ensemble.
        
        Args:
            domain_models: Dictionary mapping domains to lists of models
        """
        self.domain_models = domain_models
        self.ensembles = {}
        
        # Create ensemble for each domain
        for domain, models in domain_models.items():
            self.ensembles[domain] = AdvancedEnsemble(models, [domain])
    
    def train(self, domain_data: Dict[str, List[Tuple[List[str], str]]]):
        """
        Train domain-specific ensembles.
        
        Args:
            domain_data: Dictionary mapping domains to training data
        """
        for domain, data in domain_data.items():
            if domain in self.ensembles:
                print(f"Training ensemble for domain: {domain}")
                self.ensembles[domain].train_ensemble(data)
    
    def predict(self, context: List[str], domain: str) -> List[Tuple[str, float]]:
        """
        Make domain-specific predictions.
        
        Args:
            context: Input context
            domain: Domain identifier
            
        Returns:
            List of (word, probability) tuples
        """
        if domain not in self.ensembles:
            # Fall back to general domain
            domain = 'general'
        
        return self.ensembles[domain].predict_with_confidence(context, domain)