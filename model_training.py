import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
from models.crypto_classifier import get_training_data, SUPPORTED_ALGORITHMS

def train_models(model_choice="Auto (Ensemble)"):
    X_train, y_train = get_training_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    models = {'scaler': scaler}
    if model_choice == "Random Forest" or model_choice == "Auto (Ensemble)":
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        models['random_forest'] = rf_model
    if model_choice == "Gradient Boosting" or model_choice == "Auto (Ensemble)":
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=4,
            subsample=0.9,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        models['gradient_boosting'] = gb_model
    if model_choice == "Support Vector Machine" or model_choice == "Auto (Ensemble)":
        svm_model = SVC(
            kernel='rbf',
            C=20,
            gamma='auto',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        svm_model.fit(X_train_scaled, y_train)
        models['svm'] = svm_model
    if model_choice == "Neural Network" or model_choice == "Auto (Ensemble)":
        nn_model = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        models['neural_network'] = nn_model
    return models

def evaluate_models(models, X_test, y_test):
    X_test_scaled = models['scaler'].transform(X_test)
    evaluation = {}
    for model_name, model in models.items():
        if model_name == 'scaler':
            continue
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        evaluation[model_name] = {'accuracy': accuracy, 'confusion_matrix': conf_matrix}
    return evaluation

def predict_algorithm(features, models, confidence_threshold=0.7):
    X_scaled = models['scaler'].transform(features)
    all_predictions = {}
    for model_name, model in models.items():
        if model_name == 'scaler':
            continue
        probabilities = model.predict_proba(X_scaled)[0]
        model_predictions = dict(zip(model.classes_, probabilities))
        all_predictions[model_name] = model_predictions
    combined_predictions = {}
    model_weights = {
        'random_forest': 0.35,
        'gradient_boosting': 0.30,
        'svm': 0.20,
        'neural_network': 0.15
    }
    for algorithm in SUPPORTED_ALGORITHMS:
        weighted_sum = 0
        total_weight = 0
        for model_name, pred in all_predictions.items():
            if algorithm in pred:
                weight = model_weights.get(model_name, 0.25)
                weighted_sum += pred[algorithm] * weight
                total_weight += weight
        combined_predictions[algorithm] = weighted_sum / total_weight if total_weight > 0 else 0
    sorted_predictions = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
    top_algorithm, top_confidence = sorted_predictions[0]
    if top_confidence >= confidence_threshold:
        return top_algorithm, combined_predictions
    else:
        return None, combined_predictions
