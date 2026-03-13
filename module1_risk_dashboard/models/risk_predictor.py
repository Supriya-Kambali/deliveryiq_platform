"""
IBM DeliveryIQ — Module 1: Risk Predictor
==========================================
WHY WE USE ML HERE:
    Project risk assessment is a PREDICTION problem. Instead of a consultant
    manually guessing "this feels risky," we train Scikit-learn models on
    historical IBM project data to quantify risk with probability scores.

    This is Week 1 in action:
    - Pandas: Load and clean project data
    - Scikit-learn: Train classifiers (Random Forest, Logistic Regression)
    - NumPy: Numerical operations on feature arrays
    - Train/test split + cross-validation for model evaluation
    - Accuracy, precision, recall, F1-score for model assessment
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# WHY THESE FEATURES?
# These are the exact factors IBM delivery consultants track in
# every project. We convert consultant intuition into ML features.
# ─────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'team_size',              # Larger teams = more coordination risk
    'duration_weeks',         # Longer projects = more uncertainty
    'budget_usd',             # Higher budget = higher stakes
    'timeline_buffer_days',   # Less buffer = higher schedule risk
    'past_similar_projects',  # More experience = lower risk
    'current_week',           # How far into the project
    'tasks_completed',        # Progress tracking
    'tasks_total',            # Scope size
    'budget_spent_pct',       # Budget burn rate
    'team_experience_avg',    # Team capability score
    'complexity_encoded',     # Project complexity (encoded)
    'requirements_clarity_encoded',  # Clear requirements = lower risk
    'stakeholder_engagement_encoded' # Active stakeholders = lower risk
]

TARGET_COLUMN = 'risk_level'
RISK_LEVELS = ['Low', 'Medium', 'High', 'Critical']


class IBMRiskPredictor:
    """
    ML-powered risk predictor for IBM delivery projects.

    WHY RANDOM FOREST?
    - Handles mixed data types (numbers + categories)
    - Robust to outliers (projects vary wildly in size)
    - Provides feature importance (tells consultants WHAT is causing risk)
    - Works well with small datasets (IBM projects are limited)
    - Ensemble method = more reliable than single decision tree
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'risk_model.pkl')
        self.is_trained = False
        self.features_used = []

        # Load trained model if available
        self.load_model()

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        WHY ENCODING?
        ML models work with numbers, not text. We convert categorical
        columns like 'High complexity' → 3, 'Low complexity' → 1.
        This is Label Encoding from Week 1 preprocessing.
        """
        df = df.copy()

        # Complexity: Low=1, Medium=2, High=3, Very High=4
        complexity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
        df['complexity_encoded'] = df['complexity'].map(complexity_map).fillna(2)

        # Requirements clarity: Low=1, Medium=2, High=3
        clarity_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['requirements_clarity_encoded'] = df['requirements_clarity'].map(clarity_map).fillna(2)

        # Stakeholder engagement: Low=1, Medium=2, High=3
        df['stakeholder_engagement_encoded'] = df['stakeholder_engagement'].map(clarity_map).fillna(2)

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        WHY FEATURE ENGINEERING?
        Raw data isn't always the best input for ML. We create NEW features
        that capture deeper patterns — this is a key ML skill from Week 1.
        """
        df = df.copy()

        # Task completion rate — how much progress has been made?
        df['completion_rate'] = df['tasks_completed'] / df['tasks_total'].replace(0, 1)

        # Budget burn rate vs progress — are we overspending?
        df['budget_vs_progress'] = df['budget_spent_pct'] / (df['completion_rate'] * 100 + 1)

        # Timeline pressure — how much buffer relative to duration?
        df['timeline_pressure'] = df['timeline_buffer_days'] / df['duration_weeks']

        # Experience vs complexity — can the team handle this?
        df['experience_complexity_ratio'] = df['team_experience_avg'] / df['complexity_encoded']

        return df

    def prepare_data(self, df: pd.DataFrame):
        """Full data preparation pipeline."""
        df = self._encode_categorical(df)
        df = self._engineer_features(df)

        # Add engineered features to feature list
        all_features = FEATURE_COLUMNS + [
            'completion_rate', 'budget_vs_progress',
            'timeline_pressure', 'experience_complexity_ratio'
        ]

        # Only use features that exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        X = df[available_features].fillna(0)
        return X, available_features

    def train(self, data_path: str = None) -> dict:
        """
        Train the risk prediction model.

        WHY TRAIN/TEST SPLIT?
        We split data 80/20 to evaluate how well the model generalizes
        to NEW projects it hasn't seen — core ML evaluation from Week 1.

        WHY CROSS-VALIDATION?
        With limited IBM project data, cross-validation gives a more
        reliable estimate of model performance than a single split.
        """
        # Load data
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_projects.csv')

        df = pd.read_csv(data_path)
        print(f"📊 Loaded {len(df)} IBM project records for training")

        # Prepare features
        X, features_used = self.prepare_data(df)
        y = df[TARGET_COLUMN]

        print(f"🔧 Features used: {len(features_used)}")
        print(f"🎯 Risk distribution:\n{y.value_counts()}")

        # Train/test split — WHY 80/20? Standard ML practice, enough data to train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y) > 10 else None
        )

        # Scale features — WHY SCALING? Ensures budget ($millions) doesn't
        # dominate team_size (single digits) in the model
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest — WHY? Best balance of accuracy + interpretability
        self.model = RandomForestClassifier(
            n_estimators=100,    # 100 decision trees in the ensemble
            max_depth=10,        # Prevent overfitting
            random_state=42,     # Reproducible results
            class_weight='balanced'  # Handle imbalanced risk levels
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        # WHY THESE METRICS? From Week 1:
        # - Accuracy: Overall correctness
        # - Precision: When we say "High Risk", how often are we right?
        # - Recall: Of all actual High Risk projects, how many did we catch?
        # - F1: Balance between precision and recall
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        # Feature importance — tells consultants WHAT drives risk
        self.feature_importance = dict(zip(
            features_used,
            self.model.feature_importances_
        ))

        self.is_trained = True
        self.features_used = features_used

        print(f"\n✅ Model trained successfully!")
        print(f"   Accuracy:  {metrics['accuracy']:.2%}")
        print(f"   F1 Score:  {metrics['f1_score']:.2%}")
        print(f"\n📋 Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

        # Save model
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': features_used
        }, self.model_path)
        print(f"💾 Model saved to {self.model_path}")

        return metrics

    def load_model(self):
        """Load a previously trained model."""
        try:
            if os.path.exists(self.model_path):
                saved = joblib.load(self.model_path)
                self.model = saved.get('model')
                self.scaler = saved.get('scaler', StandardScaler())
                self.features_used = saved.get('features', [])
                if self.model and self.features_used:
                    self.is_trained = True
                    print("✅ Model loaded from disk")
                else:
                    self.is_trained = False
                    print("⚠️ Model loaded but missing required components")
            else:
                print("⚠️  No saved model found.")
                self.is_trained = False
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            self.is_trained = False

    def predict_risk(self, project_data: dict) -> dict:
        """
        Predict risk level for a new IBM project.

        Args:
            project_data: Dictionary with project attributes

        Returns:
            Dictionary with risk level, probability, and top risk factors
        """
        if not self.is_trained:
            self.load_model()

        if not self.is_trained or not hasattr(self, 'features_used') or not self.features_used:
            return {
                'risk_level': 'Unknown',
                'confidence': 0.0,
                'top_risk_factors': [],
                'recommendation': "Model not trained. Cannot provide recommendations."
            }

        # Convert to DataFrame
        df = pd.DataFrame([project_data])
        X, _ = self.prepare_data(df)

        # Ensure same features as training
        for feat in self.features_used:
            if feat not in X.columns:
                X[feat] = 0
        X = X[self.features_used].fillna(0)

        # Scale and predict
        try:
            X_scaled = self.scaler.transform(X)
            risk_level = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            classes = self.model.classes_
        except Exception as e:
            print(f"⚠️ Prediction scaling failed: {e}")
            return {
                'risk_level': 'Unknown',
                'confidence': 0.0,
                'top_risk_factors': [],
                'recommendation': "Prediction failed due to features or scaling error."
            }

        # Build probability dict
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

        # Get top risk factors (feature importance × feature value)
        feature_values = X.iloc[0].to_dict()
        risk_factors = []
        if hasattr(self, 'feature_importance') and self.feature_importance:
            for feat, importance in sorted(
                self.feature_importance.items(),
                key=lambda x: x[1], reverse=True
            )[:5]:
                if feat in feature_values:
                    risk_factors.append({
                        'factor': feat.replace('_', ' ').title(),
                        'importance': round(importance * 100, 1),
                        'value': feature_values.get(feat, 'N/A')
                    })

        return {
            'risk_level': risk_level,
            'confidence': float(max(probabilities)),
            'top_risk_factors': risk_factors,
            'recommendation': self._get_recommendation(risk_level, risk_factors)
        }

    def _get_recommendation(self, risk_level: str, risk_factors: list) -> str:
        """Generate IBM-style risk mitigation recommendation."""
        recommendations = {
            'Low': "✅ Project is on track. Maintain current delivery cadence and continue weekly status updates.",
            'Medium': "⚠️ Moderate risk detected. Schedule risk review with project sponsor. Consider adding buffer to critical path.",
            'High': "🔴 High risk — immediate action required. Escalate to IBM delivery manager. Review scope, timeline, and resource allocation.",
            'Critical': "🚨 CRITICAL — Project at serious risk of failure. Invoke IBM escalation protocol. Emergency stakeholder meeting required within 24 hours."
        }
        return recommendations.get(risk_level, "Review project status with delivery manager.")

    def get_project_health_score(self, project_data: dict) -> dict:
        """
        Calculate a 0-100 health score for the project.
        WHY A HEALTH SCORE? Consultants need a single number to communicate
        project status quickly to executives — like a patient's vital signs.
        """
        score = 100

        # Deduct points for risk factors
        if project_data.get('budget_spent_pct', 0) > 80:
            score -= 20  # Over budget
        if project_data.get('timeline_buffer_days', 10) < 3:
            score -= 25  # No timeline buffer
        if project_data.get('requirements_clarity', 'High') == 'Low':
            score -= 15  # Unclear requirements
        if project_data.get('stakeholder_engagement', 'High') == 'Low':
            score -= 10  # Poor stakeholder engagement
        if project_data.get('team_experience_avg', 4) < 3:
            score -= 15  # Inexperienced team

        # Completion rate bonus
        tasks_done = project_data.get('tasks_completed', 0)
        tasks_total = project_data.get('tasks_total', 1)
        completion_rate = tasks_done / max(tasks_total, 1)
        score += int(completion_rate * 10)

        score = max(0, min(100, score))

        # RAG status (Red/Amber/Green) — IBM standard project reporting
        if score >= 70:
            rag_status = "🟢 GREEN"
            rag_meaning = "On Track"
        elif score >= 40:
            rag_status = "🟡 AMBER"
            rag_meaning = "At Risk"
        else:
            rag_status = "🔴 RED"
            rag_meaning = "Critical"

        return {
            'health_score': score,
            'rag_status': rag_status,
            'rag_meaning': rag_meaning,
            'breakdown': {
                'Budget Health': max(0, 100 - max(0, project_data.get('budget_spent_pct', 0) - 50) * 2),
                'Timeline Health': min(100, project_data.get('timeline_buffer_days', 5) * 10),
                'Scope Health': int(completion_rate * 100),
                'Team Health': min(100, project_data.get('team_experience_avg', 3) * 20),
                'Stakeholder Health': {'High': 100, 'Medium': 60, 'Low': 20}.get(
                    project_data.get('stakeholder_engagement', 'Medium'), 60
                )
            }
        }


# ─────────────────────────────────────────────────────────────────
# DEMO: Run this file directly to see the model in action
# python risk_predictor.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("IBM DeliveryIQ — Risk Predictor Loaded")
    print("=" * 60)

    predictor = IBMRiskPredictor()

    if predictor.is_trained:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ No trained model found.")

    print("\n" + "=" * 60)
    print("🔍 Predicting risk for: IBM DeliveryIQ Final Project")
    print("=" * 60)

    my_project = {
        'team_size': 1,
        'duration_weeks': 1,
        'budget_usd': 0,
        'complexity': 'Very High',
        'requirements_clarity': 'High',
        'stakeholder_engagement': 'High',
        'timeline_buffer_days': 2,
        'past_similar_projects': 0,
        'current_week': 1,
        'tasks_completed': 3,
        'tasks_total': 9,
        'budget_spent_pct': 0,
        'team_experience_avg': 3.5
    }

    if predictor.is_trained:
        result = predictor.predict_risk(my_project)
        health = predictor.get_project_health_score(my_project)

        print(f"\n🎯 Risk Level: {result['risk_level']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"💚 Health Score: {health['health_score']}/100")
        print(f"🚦 RAG Status: {health['rag_status']} — {health['rag_meaning']}")

        print(f"\n💡 Recommendation:\n   {result['recommendation']}")

        print("\n🔝 Top Risk Factors:")
        for factor in result['top_risk_factors']:
            print(f"   • {factor['factor']}: {factor['importance']}% importance")

    else:
        print("Cannot predict — model not trained.")