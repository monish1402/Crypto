import numpy as np
import pandas as pd
from scipy import stats
import math
from collections import Counter

def extract_features(data, method="Complete Analysis (Slower)"):
    """
    Extract features from cryptographic data for algorithm identification.
    Simplified version that avoids correlation calculations to prevent errors.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Raw data to analyze
    method : str
        Feature extraction method to use
        
    Returns:
    --------
    features : numpy.ndarray
        Extracted feature vector
    feature_names : list
        Names of extracted features
    """
    # Convert data to numeric if it's not already
    if not isinstance(data[0], (int, float, np.integer, np.floating)):
        try:
            # Try to convert hex strings to bytes, directly to float64 to avoid isnan issues
            numeric_data = np.array([int(x, 16) if isinstance(x, str) and all(c in '0123456789abcdefABCDEF' for c in x) 
                                    else ord(x) if isinstance(x, str) and len(x) == 1
                                    else x for x in data], dtype=np.float64)
        except:
            # If conversion fails, use byte representation as float64
            numeric_data = np.frombuffer(str(data).encode(), dtype=np.float64)
    else:
        # Ensure we're using float64 for consistency
        numeric_data = np.array(data, dtype=np.float64)
    
    # Ensure numeric_data is a flat array
    numeric_data = np.asarray(numeric_data, dtype=np.float64).flatten()
    
    # Initialize feature lists
    features = []
    feature_names = []
    
    # SIMPLIFIED FEATURE SET THAT AVOIDS CORRELATION CALCULATIONS
    
    # 1. Basic statistics - always compute these
    # Mean and standard deviation
    mean_val = float(np.mean(numeric_data))
    features.append(mean_val)
    feature_names.append("mean")
    
    std_val = float(np.std(numeric_data))
    features.append(std_val)
    feature_names.append("std")
    
    # Skewness and kurtosis (safely handled)
    try:
        if len(numeric_data) > 2:
            skewness = float(stats.skew(numeric_data))
        else:
            skewness = 0.0
    except:
        skewness = 0.0
    features.append(skewness)
    feature_names.append("skewness")
    
    try:
        if len(numeric_data) > 3:
            kurtosis = float(stats.kurtosis(numeric_data))
        else:
            kurtosis = 0.0
    except:
        kurtosis = 0.0
    features.append(kurtosis)
    feature_names.append("kurtosis")
    
    # 2. Quartile statistics
    try:
        q1, median, q3 = np.percentile(numeric_data, [25, 50, 75])
        q1, median, q3 = float(q1), float(median), float(q3)
    except:
        q1, median, q3 = 0.0, 0.0, 0.0
    
    features.extend([q1, median, q3])
    feature_names.extend(["q1", "median", "q3"])
    
    # IQR
    iqr = q3 - q1
    features.append(iqr)
    feature_names.append("iqr")
    
    # Range
    try:
        data_range = float(np.max(numeric_data) - np.min(numeric_data))
    except:
        data_range = 0.0
    features.append(data_range)
    feature_names.append("range")
    
    # 3. Histogram-based features (distribution shape)
    try:
        # Create histogram with fixed bins
        hist, _ = np.histogram(numeric_data, bins=10, density=True)
        # Add histogram bin values as features
        for i, bin_val in enumerate(hist):
            features.append(float(bin_val))
            feature_names.append(f"hist_bin_{i}")
    except:
        # Add zeros if histogram calculation fails
        for i in range(10):
            features.append(0.0)
            feature_names.append(f"hist_bin_{i}")
    
    # 4. Entropy-based features
    try:
        # Safely convert to integers in 0-255 range
        numeric_int = np.clip(np.round(numeric_data), 0, 255).astype(np.int32)
        
        # Calculate byte frequencies using safe integers
        byte_counts = np.bincount(numeric_int, minlength=256)
        probabilities = byte_counts / len(numeric_data)
        
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        entropy = float(entropy)
        
        # Normalized entropy (0-1 scale)
        max_entropy = math.log2(256)  # Maximum entropy for byte values
        norm_entropy = entropy / max_entropy
    except Exception as e:
        print(f"Entropy calculation error: {str(e)}")
        entropy = 0.0
        norm_entropy = 0.0
    
    features.append(entropy)
    feature_names.append("shannon_entropy")
    
    features.append(norm_entropy)
    feature_names.append("normalized_entropy")
    
    # 5. Byte frequency analysis
    try:
        # Use the same numeric_int from entropy calculation
        numeric_int = np.clip(np.round(numeric_data), 0, 255).astype(np.int32)
        
        # Calculate byte frequencies
        byte_counts = np.bincount(numeric_int, minlength=256)
        
        # Most common byte frequency
        most_common_byte = byte_counts.argmax()
        most_common_freq = byte_counts[most_common_byte] / len(numeric_data)
        
        # Calculate probabilities for each byte
        probabilities = byte_counts / len(numeric_data)
        
        # Byte distribution uniformity (using chi-squared distance from uniform)
        uniform_prob = 1/256
        uniform_distance = np.sum((probabilities - uniform_prob)**2) / uniform_prob
    except Exception as e:
        print(f"Byte frequency analysis error: {str(e)}")
        most_common_freq = 0.0
        uniform_distance = 0.0
    
    features.append(float(most_common_freq))
    feature_names.append("most_common_byte_freq")
    
    features.append(float(uniform_distance))
    feature_names.append("uniformity_chi2")
    
    # 6. Pattern-based features
    # Count repeating bytes
    try:
        # Count of repeated adjacent bytes
        repeated_bytes = sum(1 for i in range(len(numeric_data)-1) if numeric_data[i] == numeric_data[i+1])
        repeat_ratio = repeated_bytes / (len(numeric_data)-1) if len(numeric_data) > 1 else 0
    except:
        repeat_ratio = 0.0
    
    features.append(float(repeat_ratio))
    feature_names.append("repeat_ratio")
    
    # Ensure we always return exactly 30 features to match the training data
    expected_features = 30
    if len(features) < expected_features:
        # Pad with zeros if we have fewer features than expected
        features.extend([0.0] * (expected_features - len(features)))
        feature_names.extend([f"padding_{i}" for i in range(len(feature_names), expected_features)])
    elif len(features) > expected_features:
        # Truncate if we have more features than expected
        features = features[:expected_features]
        feature_names = feature_names[:expected_features]
    
    return np.array(features).reshape(1, -1), feature_names


def get_feature_importance(models, feature_names):
    """
    Extract feature importance from trained models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    feature_names : list
        Names of features
    
    Returns:
    --------
    importance : dict
        Dictionary mapping feature names to their importance
    """
    importance = {}
    
    # Try to get feature importance from models that support it
    if 'random_forest' in models:
        rf_importance = models['random_forest'].feature_importances_
        for i, name in enumerate(feature_names):
            importance[name] = rf_importance[i]
    elif 'gradient_boosting' in models:
        gb_importance = models['gradient_boosting'].feature_importances_
        for i, name in enumerate(feature_names):
            importance[name] = gb_importance[i]
    else:
        # If no model with feature_importances_, use a placeholder
        for name in feature_names:
            importance[name] = 1.0 / len(feature_names)
    
    return importance
