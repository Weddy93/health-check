import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

class EnhancedConfidentMalnutritionClassifier:
    """Enhanced classifier with confidence scoring for malnutrition detection"""

    def __init__(self, base_model=None):
        if base_model is None:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
        else:
            self.model = base_model
        self.confidence_threshold = 0.8

    def fit(self, X, y):
        """Fit the model"""
        self.model.fit(X, y)
        return self

    def calculate_confidence(self, features, prediction_proba):
        """Calculate confidence score"""
        # Base confidence from model probability
        max_prob = np.max(prediction_proba)

        # Feature reliability score (check for outliers)
        feature_reliability = self._check_feature_reliability(features)

        # Combine scores
        confidence = (max_prob * 0.7) + (feature_reliability * 0.3)

        return confidence

    def _check_feature_reliability(self, features):
        """Check if features are within expected ranges"""
        reliability_scores = []

        # Check z-scores are within reasonable ranges
        for col in features.columns:
            if 'z' in col:
                z_scores = features[col]
                # Give lower reliability for extreme z-scores
                reliability = 1 - (np.abs(z_scores) > 3).mean()
                reliability_scores.append(reliability)

        return np.mean(reliability_scores) if reliability_scores else 0.5

    def predict_with_confidence(self, X):
        """Make predictions with confidence scores"""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        confidences = [self.calculate_confidence(X.iloc[i:i+1], prob)
                      for i, prob in enumerate(probabilities)]

        return predictions, confidences

    def predict(self, X):
        """Standard predict method"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Standard predict_proba method"""
        return self.model.predict_proba(X)
