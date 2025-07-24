# ml_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class FeatureTransformer:
    def fit_transform(self, X, y=None):
        self.vec = TfidfVectorizer()
        symptom_vecs = self.vec.fit_transform(X['symptoms'].str.lower())
        numeric = X[['age', 'gender']].values
        from scipy.sparse import hstack
        return hstack((symptom_vecs, numeric))

    def transform(self, X):
        symptom_vecs = self.vec.transform(X['symptoms'].str.lower())
        numeric = X[['age', 'gender']].values
        from scipy.sparse import hstack
        return hstack((symptom_vecs, numeric))

class MultiOutputModel:
    def __init__(self):
        self.features = FeatureTransformer()
        self.disease_model = RandomForestClassifier()
        self.severity_model = RandomForestClassifier()
        self.confidence_model = RandomForestClassifier()

    def fit(self, X, y):
        X_trans = self.features.fit_transform(X)
        self.disease_model.fit(X_trans, y['disease'])
        self.severity_model.fit(X_trans, y['severity'])
        self.confidence_model.fit(X_trans, y['confidence'])

    def predict(self, X):
        X_trans = self.features.transform(X)
        return {
            'disease': self.disease_model.predict(X_trans)[0],
            'severity': self.severity_model.predict(X_trans)[0],
            'confidence': self.confidence_model.predict(X_trans)[0]
        }
