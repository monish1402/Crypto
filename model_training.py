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
    """
    Train machine learning models for cryptographic algorithm identification.
    
    Parameters:
    -----------
    model_choice : str
        The model type to train and use for prediction
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    """
    # Get training data
    X_train, y_train = get_training_data()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {'scaler': scaler}
    
    # Train selected model(s)
    if model_choice == "Random Forest" or model_choice == "Auto (Ensemble)":
        rf_model = RandomForestClassifier(
            n_estimators=200,      # Increased from 100 to 200
            max_depth=25,          # Increased from 20 to 25
            min_samples_split=4,   # Reduced from 5 to 4 for more detailed trees
            min_samples_leaf=2,    # Added parameter for better generalization
            class_weight='balanced',# Added to handle any class imbalance
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        models['random_forest'] = rf_model
    
    if model_choice == "Gradient Boosting" or model_choice == "Auto (Ensemble)":
        gb_model = GradientBoostingClassifier(
            n_estimators=150,      # Increased from 100 to 150
            learning_rate=0.05,    # Reduced from 0.1 to 0.05 for better generalization
            max_depth=6,           # Increased from 5 to 6
            min_samples_split=4,   # Added parameter
            subsample=0.9,         # Added parameter to reduce overfitting
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        models['gradient_boosting'] = gb_model
    
    if model_choice == "Support Vector Machine" or model_choice == "Auto (Ensemble)":
        svm_model = SVC(
            kernel='rbf',
            C=20,                 # Increased from 10 to 20
            gamma='auto',         # Changed from 'scale' to 'auto'
            probability=True,
            class_weight='balanced',# Added to handle any class imbalance
            random_state=42
        )
        svm_model.fit(X_train_scaled, y_train)
        models['svm'] = svm_model
    
    if model_choice == "Neural Network" or model_choice == "Auto (Ensemble)":
        nn_model = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),  # Increased network depth and width
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',      # Changed to adaptive learning rate
            max_iter=1000,                 # Increased from 500 to 1000
            early_stopping=True,           # Added early stopping
            validation_fraction=0.1,       # Added validation split
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        models['neural_network'] = nn_model
    
    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : numpy.ndarray
        Test feature data
    y_test : numpy.ndarray
        Test labels
        
    Returns:
    --------
    evaluation : dict
        Dictionary containing evaluation metrics
    """
    X_test_scaled = models['scaler'].transform(X_test)
    evaluation = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        if model_name == 'scaler':
            continue
            
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        evaluation[model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
    
    return evaluation


def predict_algorithm(features, models, confidence_threshold=0.7):
    """
    Predict the cryptographic algorithm used in the given data.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Extracted features from the data
    models : dict
        Dictionary of trained models
    confidence_threshold : float
        Minimum confidence level to report a match
        
    Returns:
    --------
    predicted_algorithm : str
        Name of the identified algorithm
    confidence_scores : dict
        Confidence scores for each algorithm
    """
    # Scale features
    X_scaled = models['scaler'].transform(features)
    
    all_predictions = {}
    
    # Make predictions with each model
    for model_name, model in models.items():
        if model_name == 'scaler':
            continue
        
        # Get probability predictions for each class
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Map probabilities to algorithm names
        model_predictions = dict(zip(model.classes_, probabilities))
        
        # Store predictions for this model
        all_predictions[model_name] = model_predictions
    
    # Combine predictions from all models (weighted ensemble)
    combined_predictions = {}
    
    # Define weights for each model (giving higher weight to more accurate models)
    model_weights = {
        'random_forest': 0.35,      # Higher weight for Random Forest 
        'gradient_boosting': 0.30,  # Good weight for Gradient Boosting
        'svm': 0.20,                # Medium weight for SVM
        'neural_network': 0.15      # Lower weight for Neural Network
    }
    
    for algorithm in SUPPORTED_ALGORITHMS:
        # Calculate weighted average confidence score for each algorithm
        weighted_sum = 0
        total_weight = 0
        
        for model_name, pred in all_predictions.items():
            if algorithm in pred:
                weight = model_weights.get(model_name, 0.25)  # Default weight if not specified
                weighted_sum += pred[algorithm] * weight
                total_weight += weight
        
        # Calculate weighted average
        combined_predictions[algorithm] = weighted_sum / total_weight if total_weight > 0 else 0
    
    # Get the algorithm with highest confidence
    sorted_predictions = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
    top_algorithm, top_confidence = sorted_predictions[0]
    
    # Return prediction only if confidence exceeds threshold
    if top_confidence >= confidence_threshold:
        return top_algorithm, combined_predictions
    else:
        return None, combined_predictions
