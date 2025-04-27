import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# List of supported cryptographic algorithms for identification
SUPPORTED_ALGORITHMS = [
    "AES", "DES", "3DES", "RC4", "RSA", 
    "Blowfish", "SHA-256", "MD5"
]

def generate_synthetic_features():
    """
    Generate synthetic features for training the model.
    This is used when no real dataset is available.
    
    Returns:
    --------
    X : numpy.ndarray
        Generated feature matrix
    y : numpy.ndarray
        Generated labels
    """
    # Number of samples to generate per algorithm (increased for better training)
    samples_per_algorithm = 300  # Increased from 100 to 300
    total_samples = samples_per_algorithm * len(SUPPORTED_ALGORITHMS)
    
    # Number of features
    n_features = 30
    
    # Initialize data arrays
    X = np.zeros((total_samples, n_features))
    y = np.zeros(total_samples, dtype=object)
    
    # Generate synthetic features for each algorithm
    for i, algorithm in enumerate(SUPPORTED_ALGORITHMS):
        start_idx = i * samples_per_algorithm
        end_idx = (i + 1) * samples_per_algorithm
        
        # Base feature parameters for each algorithm
        if algorithm == "AES":
            # High entropy, block size features, more uniform distribution
            base_features = np.array([
                0.998,  # Shannon entropy (high)
                0.999,  # Normalized entropy (very high)
                0.002,  # Entropy deviation (very low)
                0.005,  # Entropy consistency (very consistent)
                0.001,  # Block correlation for 16 bytes (very low)
                0.002,  # Block correlation for 32 bytes (very low)
                0.125,  # Mean (expected for uniform bytes)
                0.7,    # Std
                0.02,   # Skewness (close to 0)
                -0.01,  # Kurtosis (close to 0)
                0.0625, # Q1
                0.125,  # Median
                0.1875, # Q3
                0.125,  # IQR
                0.998,  # Range (full range of bytes)
            ] + [0.004] * 15)  # Remaining features (frequency patterns, etc.)
            
        elif algorithm == "DES":
            # High entropy, specific frequency features
            base_features = np.array([
                0.97,   # Shannon entropy (high but less than AES)
                0.98,   # Normalized entropy (high)
                0.02,   # Entropy deviation (low)
                0.01,   # Entropy consistency (consistent)
                0.02,   # Block correlation for 8 bytes (low)
                0.01,   # Block correlation for 16 bytes (low)
                0.125,  # Mean
                0.72,   # Std
                0.03,   # Skewness
                -0.02,  # Kurtosis
                0.062,  # Q1
                0.125,  # Median
                0.188,  # Q3
                0.126,  # IQR
                0.99,   # Range
            ] + [0.005] * 15)  # Remaining features
            
        elif algorithm == "3DES":
            # Similar to DES but higher entropy
            base_features = np.array([
                0.985,  # Shannon entropy 
                0.99,   # Normalized entropy
                0.01,   # Entropy deviation
                0.008,  # Entropy consistency
                0.015,  # Block correlation for 8 bytes
                0.007,  # Block correlation for 16 bytes
                0.125,  # Mean
                0.71,   # Std
                0.025,  # Skewness
                -0.015, # Kurtosis
                0.062,  # Q1
                0.125,  # Median
                0.188,  # Q3
                0.126,  # IQR
                0.995,  # Range
            ] + [0.0045] * 15)  # Remaining features
            
        elif algorithm == "RC4":
            # Stream cipher characteristics
            base_features = np.array([
                0.99,   # Shannon entropy (very high)
                0.995,  # Normalized entropy (very high)
                0.005,  # Entropy deviation (very low)
                0.003,  # Entropy consistency (very consistent)
                0.1,    # Block correlation for 8 bytes (higher than block ciphers)
                0.08,   # Block correlation for 16 bytes (higher than block ciphers)
                0.125,  # Mean
                0.705,  # Std
                0.01,   # Skewness
                -0.005, # Kurtosis
                0.063,  # Q1
                0.125,  # Median
                0.187,  # Q3
                0.124,  # IQR
                0.999,  # Range
            ] + [0.004] * 15)  # Remaining features
            
        elif algorithm == "RSA":
            # Asymmetric cipher characteristics (longer blocks)
            base_features = np.array([
                0.96,   # Shannon entropy
                0.97,   # Normalized entropy
                0.03,   # Entropy deviation
                0.02,   # Entropy consistency
                0.15,   # Block correlation for 64 bytes (higher)
                0.12,   # Block correlation for 128 bytes (higher)
                0.13,   # Mean
                0.74,   # Std
                0.05,   # Skewness
                -0.04,  # Kurtosis
                0.065,  # Q1
                0.13,   # Median
                0.195,  # Q3
                0.13,   # IQR
                0.98,   # Range
            ] + [0.006] * 15)  # Remaining features
            
        elif algorithm == "Blowfish":
            # Block cipher with distinctive patterns
            base_features = np.array([
                0.975,  # Shannon entropy
                0.985,  # Normalized entropy
                0.015,  # Entropy deviation
                0.01,   # Entropy consistency
                0.008,  # Block correlation for 8 bytes
                0.006,  # Block correlation for 16 bytes
                0.125,  # Mean
                0.715,  # Std
                0.025,  # Skewness
                -0.015, # Kurtosis
                0.062,  # Q1
                0.125,  # Median
                0.188,  # Q3
                0.126,  # IQR
                0.995,  # Range
            ] + [0.005] * 15)  # Remaining features
            
        elif algorithm == "SHA-256":
            # Hash function characteristics
            base_features = np.array([
                0.999,  # Shannon entropy (extremely high)
                0.9999, # Normalized entropy (extremely high)
                0.0001, # Entropy deviation (extremely low)
                0.0001, # Entropy consistency (extremely consistent)
                0.0005, # Block correlation for 32 bytes (extremely low)
                0.0004, # Block correlation for 64 bytes (extremely low)
                0.125,  # Mean
                0.7,    # Std
                0.001,  # Skewness
                -0.001, # Kurtosis
                0.0625, # Q1
                0.125,  # Median
                0.1875, # Q3
                0.125,  # IQR
                0.9999, # Range
            ] + [0.0038] * 15)  # Remaining features
            
        elif algorithm == "MD5":
            # Hash function characteristics (but less uniform than SHA-256)
            base_features = np.array([
                0.995,  # Shannon entropy (very high)
                0.997,  # Normalized entropy (very high)
                0.003,  # Entropy deviation (very low)
                0.002,  # Entropy consistency (very consistent)
                0.001,  # Block correlation for 16 bytes (very low)
                0.0008, # Block correlation for 32 bytes (very low)
                0.125,  # Mean
                0.705,  # Std
                0.004,  # Skewness
                -0.003, # Kurtosis
                0.0625, # Q1
                0.125,  # Median
                0.1875, # Q3
                0.125,  # IQR
                0.998,  # Range
            ] + [0.004] * 15)  # Remaining features
        
        # Generate samples with noise for this algorithm
        for j in range(samples_per_algorithm):
            sample_idx = start_idx + j
            
            # Add random noise to base features (with reduced noise for better accuracy)
            noise = np.random.normal(0, 0.01, n_features)  # Reduced from 0.02 to 0.01
            
            # Ensure features stay in valid ranges after adding noise
            features = np.clip(base_features + noise, 0, 1)
            
            # Store features and label
            X[sample_idx] = features[:n_features]  # Ensure we only use n_features
            y[sample_idx] = algorithm
    
    return X, y


def get_training_data():
    """
    Get or generate training data for the cryptographic algorithm classifier.
    
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target labels
    """
    # For now, use synthetic data (would be replaced with real data when available)
    X, y = generate_synthetic_features()
    
    return X, y
