import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

def visualize_data_distribution(data, title="Data Distribution"):
    """
    Create a visualization of the data distribution.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to visualize
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ensure data is flattened
    flat_data = data.flatten()
    
    # Histogram
    ax1.hist(flat_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Byte Value Distribution Histogram')
    ax1.set_xlabel('Byte Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Calculate running entropy for visualization
    window_size = min(1000, len(flat_data))
    if len(flat_data) > window_size:
        entropy_values = []
        for i in range(0, len(flat_data) - window_size, window_size // 10):
            window = flat_data[i:i+window_size]
            # Count frequencies of each byte value in the window
            counts = Counter(window.astype(np.uint8))
            # Calculate probabilities
            probabilities = [count / window_size for count in counts.values()]
            # Calculate entropy
            entropy = -sum(p * np.log2(p) for p in probabilities)
            entropy_values.append(entropy)
        
        # Plot entropy variation
        ax2.plot(entropy_values, color='green')
        ax2.set_title('Running Entropy Analysis')
        ax2.set_xlabel('Window Index')
        ax2.set_ylabel('Shannon Entropy')
        ax2.grid(True, alpha=0.3)
    else:
        # If data is too small, show a different visualization
        ax2.hexbin(np.arange(len(flat_data)), flat_data, gridsize=25, cmap='viridis')
        ax2.set_title('Byte Value Pattern Visualization')
        ax2.set_xlabel('Byte Position')
        ax2.set_ylabel('Byte Value')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f"Analysis of {title}", y=1.05)
    
    return fig

def plot_feature_importance(importance_dict):
    """
    Create a visualization of feature importance.
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary mapping feature names to importance values
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    """
    # Sort features by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 15 features for better visualization
    features = [item[0] for item in sorted_features[:15]]
    importances = [item[1] for item in sorted_features[:15]]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center', color='skyblue', edgecolor='black')
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:20] for f in features])  # Truncate long feature names
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importance for Algorithm Identification')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(cm, class_names):
    """
    Create a visualization of the confusion matrix.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    class_names : list
        Names of classes
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax)
    
    # Set labels and title
    ax.set_ylabel('True Algorithm')
    ax.set_xlabel('Predicted Algorithm')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    return fig

def plot_algorithm_confidence(confidence_scores):
    """
    Create a visualization of algorithm confidence scores.
    
    Parameters:
    -----------
    confidence_scores : dict
        Dictionary mapping algorithm names to confidence scores
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    """
    # Sort algorithms by confidence score
    sorted_algorithms = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    
    algorithms = [item[0] for item in sorted_algorithms]
    scores = [item[1] for item in sorted_algorithms]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart with custom colors
    y_pos = np.arange(len(algorithms))
    bars = ax.barh(y_pos, scores, align='center', edgecolor='black')
    
    # Color coding based on confidence level
    for i, bar in enumerate(bars):
        if scores[i] >= 0.8:
            bar.set_color('green')
        elif scores[i] >= 0.5:
            bar.set_color('skyblue')
        else:
            bar.set_color('lightgray')
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(algorithms)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Confidence Score')
    ax.set_title('Algorithm Identification Confidence')
    
    # Add vertical line at common threshold values
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='High Confidence')
    
    # Add legend
    ax.legend()
    
    # Set x-axis limits
    ax.set_xlim(0, 1.0)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    return fig
