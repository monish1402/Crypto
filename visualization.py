import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

def visualize_data_distribution(data, title="Data Distribution"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    flat_data = data.flatten()
    ax1.hist(flat_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Byte Value Distribution Histogram')
    ax1.set_xlabel('Byte Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    window_size = min(1000, len(flat_data))
    if len(flat_data) > window_size:
        entropy_values = []
        for i in range(0, len(flat_data) - window_size, window_size // 10):
            window = flat_data[i:i+window_size]
            counts = Counter(window.astype(np.uint8))
            probabilities = [count / window_size for count in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities)
            entropy_values.append(entropy)
        ax2.plot(entropy_values, color='green')
        ax2.set_title('Running Entropy Analysis')
        ax2.set_xlabel('Window Index')
        ax2.set_ylabel('Shannon Entropy')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.hexbin(np.arange(len(flat_data)), flat_data, gridsize=25, cmap='viridis')
        ax2.set_title('Byte Value Pattern Visualization')
        ax2.set_xlabel('Byte Position')
        ax2.set_ylabel('Byte Value')
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle(f"Analysis of {title}", y=1.05)
    return fig

def plot_feature_importance(importance_dict):
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_features[:15]]
    importances = [item[1] for item in sorted_features[:15]]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, align='center', color='skyblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:20] for f in features])
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importance for Algorithm Identification')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel('True Algorithm')
    ax.set_xlabel('Predicted Algorithm')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

def plot_algorithm_confidence(confidence_scores):
    sorted_algorithms = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    algorithms = [item[0] for item in sorted_algorithms]
    scores = [item[1] for item in sorted_algorithms]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(algorithms))
    bars = ax.barh(y_pos, scores, align='center', edgecolor='black')
    for i, bar in enumerate(bars):
        if scores[i] >= 0.8:
            bar.set_color('green')
        elif scores[i] >= 0.5:
            bar.set_color('skyblue')
        else:
            bar.set_color('lightgray')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(algorithms)
    ax.invert_yaxis()
    ax.set_xlabel('Confidence Score')
    ax.set_title('Algorithm Identification Confidence')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='High Confidence')
    ax.legend()
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    return fig
