import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import time
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="Cryptographic Algorithm Identifier",
    page_icon="🔐",
    layout="wide"
)

# App title and description
st.title("Cryptographic Algorithm Identifier")
st.markdown("""
This application uses pattern analysis to identify cryptographic algorithms used in data files.
Upload your cryptographic data to get started.
""")

# List of supported algorithms
SUPPORTED_ALGORITHMS = [
    "AES", "DES", "3DES", "RC4", "RSA", 
    "Blowfish", "SHA-256", "MD5"
]

# Create sidebar for options
st.sidebar.header("Settings")

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Match Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
    step=0.05,
    help="Minimum confidence level to report an algorithm match"
)

# File uploader
st.subheader("Upload Cryptographic Data")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "bin", "hex"])

# Sample format information
with st.expander("Supported File Formats"):
    st.markdown("""
    - **CSV**: Comma-separated values with binary or hex data
    - **TXT**: Plain text files with cryptographic outputs
    - **BIN**: Binary files containing encrypted data
    - **HEX**: Hexadecimal representation of encrypted data
    
    Supported algorithms for identification include:
    """ + ", ".join(SUPPORTED_ALGORITHMS))

# Function to extract simple patterns from binary data
def extract_patterns(data):
    # Make sure data is treated as bytes
    byte_data = np.frombuffer(data, dtype=np.uint8)
    
    # Store the patterns
    patterns = {}
    
    # 1. Byte frequency distribution
    frequencies = Counter(byte_data)
    total_bytes = len(byte_data)
    
    # Calculate frequency distribution
    byte_distribution = {byte: count / total_bytes for byte, count in frequencies.items()}
    patterns['byte_distribution'] = byte_distribution
    
    # 2. Basic statistics that are safe to calculate
    patterns['mean'] = float(np.mean(byte_data))
    patterns['median'] = float(np.median(byte_data))
    
    # 3. Calculate repeating patterns
    repeats = 0
    for i in range(len(byte_data) - 1):
        if byte_data[i] == byte_data[i + 1]:
            repeats += 1
    patterns['repeat_rate'] = repeats / (len(byte_data) - 1) if len(byte_data) > 1 else 0
    
    # 4. Check for block patterns (common in block ciphers)
    block_sizes = [8, 16, 32, 64, 128]
    block_patterns = {}
    
    for size in block_sizes:
        if len(byte_data) >= size * 2:
            # Count similar blocks
            similar_blocks = 0
            total_blocks = len(byte_data) // size - 1
            
            for i in range(total_blocks):
                block1 = byte_data[i * size:(i + 1) * size]
                block2 = byte_data[(i + 1) * size:(i + 2) * size]
                
                # Calculate simple similarity (count matching bytes)
                matches = sum(1 for a, b in zip(block1, block2) if a == b)
                if matches / size > 0.5:  # If more than 50% similar
                    similar_blocks += 1
            
            if total_blocks > 0:
                block_patterns[size] = similar_blocks / total_blocks
            else:
                block_patterns[size] = 0
    
    patterns['block_patterns'] = block_patterns
    
    # 5. Entropy estimation (simplified and safer)
    unique_bytes = len(frequencies)
    patterns['unique_byte_ratio'] = unique_bytes / 256  # How many of the possible byte values are used
    
    # Get most common byte and its frequency
    if frequencies:
        most_common = frequencies.most_common(1)[0]
        patterns['most_common_byte'] = most_common[0]
        patterns['most_common_frequency'] = most_common[1] / total_bytes
    else:
        patterns['most_common_byte'] = 0
        patterns['most_common_frequency'] = 0
        
    return patterns

# Function to visualize data patterns
def visualize_patterns(patterns, title="Data Patterns"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot byte distribution
    byte_vals = list(patterns['byte_distribution'].keys())
    byte_freqs = list(patterns['byte_distribution'].values())
    
    if byte_vals and byte_freqs:
        ax1.bar(byte_vals, byte_freqs, alpha=0.7, color='skyblue')
        ax1.set_title('Byte Value Distribution')
        ax1.set_xlabel('Byte Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No byte data available", ha='center', va='center')
    
    # Plot block similarity patterns
    if patterns['block_patterns']:
        block_sizes = list(patterns['block_patterns'].keys())
        similarities = list(patterns['block_patterns'].values())
        
        ax2.bar(block_sizes, similarities, alpha=0.7, color='green')
        ax2.set_title('Block Size Similarity Patterns')
        ax2.set_xlabel('Block Size (bytes)')
        ax2.set_ylabel('Similarity Ratio')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Block pattern analysis not available", ha='center', va='center')
    
    plt.tight_layout()
    plt.suptitle(f"Analysis of {title}", y=1.05)
    
    return fig

# Function to match pattern to known algorithms
def match_algorithm(patterns):
    # Simplified matching based on known algorithm characteristics
    confidence_scores = {}
    
    # AES characteristics (high entropy, block size 16 bytes)
    aes_score = 0.0
    if patterns['unique_byte_ratio'] > 0.9:  # High entropy
        aes_score += 0.4
    if patterns['block_patterns'].get(16, 0) < 0.2:  # Low similarity in 16-byte blocks
        aes_score += 0.4
    if patterns['most_common_frequency'] < 0.02:  # No dominant byte
        aes_score += 0.2
    confidence_scores['AES'] = aes_score
    
    # DES characteristics (block size 8 bytes)
    des_score = 0.0
    if patterns['unique_byte_ratio'] > 0.8:  # High entropy but less than AES
        des_score += 0.3
    if patterns['block_patterns'].get(8, 0) < 0.25:  # Low similarity in 8-byte blocks
        des_score += 0.5
    if patterns['repeat_rate'] < 0.1:  # Few repeats
        des_score += 0.2
    confidence_scores['DES'] = des_score
    
    # 3DES (similar to DES but higher entropy)
    des3_score = 0.0
    if patterns['unique_byte_ratio'] > 0.85:  # Higher entropy than DES
        des3_score += 0.4
    if patterns['block_patterns'].get(8, 0) < 0.2:  # Lower block similarity than DES
        des3_score += 0.4
    if patterns['repeat_rate'] < 0.05:  # Very few repeats
        des3_score += 0.2
    confidence_scores['3DES'] = des3_score
    
    # RC4 (stream cipher - more uniform distribution)
    rc4_score = 0.0
    if patterns['unique_byte_ratio'] > 0.95:  # Very high entropy
        rc4_score += 0.5
    if patterns['most_common_frequency'] < 0.015:  # Very flat distribution
        rc4_score += 0.3
    if sum(patterns['block_patterns'].values()) / len(patterns['block_patterns']) > 0.2:  # No strong block patterns
        rc4_score += 0.2
    confidence_scores['RC4'] = rc4_score
    
    # RSA characteristics (large block size, some structure)
    rsa_score = 0.0
    if patterns['unique_byte_ratio'] > 0.7:  # Good entropy but not as high as symmetric
        rsa_score += 0.3
    if patterns['block_patterns'].get(64, 0) > 0.1 or patterns['block_patterns'].get(128, 0) > 0.1:
        rsa_score += 0.4  # Some patterns at larger block sizes
    if 0.01 < patterns['most_common_frequency'] < 0.05:  # Some structure but not too much
        rsa_score += 0.3
    confidence_scores['RSA'] = rsa_score
    
    # Blowfish characteristics
    bf_score = 0.0
    if patterns['unique_byte_ratio'] > 0.88:  # High entropy
        bf_score += 0.4
    if patterns['block_patterns'].get(8, 0) < 0.22:  # Low similarity in 8-byte blocks
        bf_score += 0.4
    if patterns['repeat_rate'] < 0.08:  # Few repeats
        bf_score += 0.2
    confidence_scores['Blowfish'] = bf_score
    
    # SHA-256 (very high entropy, no obvious patterns)
    sha256_score = 0.0
    if patterns['unique_byte_ratio'] > 0.99:  # Extremely high entropy
        sha256_score += 0.6
    if patterns['most_common_frequency'] < 0.01:  # Almost perfectly flat distribution
        sha256_score += 0.3
    if patterns['repeat_rate'] < 0.01:  # Almost no repeats
        sha256_score += 0.1
    confidence_scores['SHA-256'] = sha256_score
    
    # MD5 (high entropy but not as high as SHA-256)
    md5_score = 0.0
    if patterns['unique_byte_ratio'] > 0.98:  # Very high entropy
        md5_score += 0.5
    if patterns['most_common_frequency'] < 0.015:  # Very flat distribution
        md5_score += 0.3
    if patterns['repeat_rate'] < 0.02:  # Very few repeats
        md5_score += 0.2
    confidence_scores['MD5'] = md5_score
    
    # Find the best match
    best_algorithm = max(confidence_scores, key=confidence_scores.get)
    highest_confidence = confidence_scores[best_algorithm]
    
    if highest_confidence >= confidence_threshold:
        return best_algorithm, confidence_scores
    else:
        return None, confidence_scores

# Function to plot confidence scores
def plot_confidence_scores(confidence_scores):
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

# Main analysis section
if uploaded_file is not None:
    # Display file details
    file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size} bytes"}
    st.write(file_details)

    try:
        # Read the file content
        content = uploaded_file.read()
        
        with st.spinner("Analyzing cryptographic data..."):
            # Extract patterns from the data
            patterns = extract_patterns(content)
            
            # Display data patterns
            st.subheader("Data Pattern Analysis")
            fig1 = visualize_patterns(patterns, uploaded_file.name)
            st.pyplot(fig1)
            
            # Match algorithm
            predicted_algorithm, confidence_scores = match_algorithm(patterns)
            
            # Display results
            st.subheader("Cryptographic Algorithm Identification Results")
            
            if predicted_algorithm:
                results_col1, results_col2 = st.columns([2, 1])
                
                with results_col1:
                    st.success(f"**Identified Algorithm: {predicted_algorithm}**")
                    
                    fig2 = plot_confidence_scores(confidence_scores)
                    st.pyplot(fig2)
                
                with results_col2:
                    st.markdown("### Algorithm Properties")
                    
                    algorithm_info = {
                        "AES": {
                            "Type": "Symmetric Block Cipher",
                            "Key Sizes": "128, 192, or 256 bits",
                            "Block Size": "128 bits",
                            "Structure": "Substitution-Permutation Network"
                        },
                        "DES": {
                            "Type": "Symmetric Block Cipher",
                            "Key Size": "56 bits",
                            "Block Size": "64 bits",
                            "Structure": "Feistel Network"
                        },
                        "3DES": {
                            "Type": "Symmetric Block Cipher",
                            "Key Size": "168 bits (effective: 112 bits)",
                            "Block Size": "64 bits",
                            "Structure": "Feistel Network"
                        },
                        "RC4": {
                            "Type": "Symmetric Stream Cipher",
                            "Key Size": "40-2048 bits",
                            "State Size": "2048 bits",
                            "Structure": "Stream Cipher"
                        },
                        "RSA": {
                            "Type": "Asymmetric Cipher",
                            "Key Size": "1024-4096 bits",
                            "Security": "Based on factoring problem",
                            "Structure": "Public-key cryptosystem"
                        },
                        "Blowfish": {
                            "Type": "Symmetric Block Cipher",
                            "Key Size": "32-448 bits",
                            "Block Size": "64 bits",
                            "Structure": "Feistel Network"
                        },
                        "SHA-256": {
                            "Type": "Cryptographic Hash Function",
                            "Output Size": "256 bits",
                            "Internal State": "256 bits",
                            "Structure": "Merkle–Damgård construction"
                        },
                        "MD5": {
                            "Type": "Cryptographic Hash Function",
                            "Output Size": "128 bits",
                            "Internal State": "128 bits",
                            "Structure": "Merkle–Damgård construction"
                        }
                    }
                    
                    if predicted_algorithm in algorithm_info:
                        for key, value in algorithm_info[predicted_algorithm].items():
                            st.markdown(f"**{key}**: {value}")
                    else:
                        st.info("Detailed information not available for this algorithm.")
            else:
                st.warning("Could not confidently identify the cryptographic algorithm. Try adjusting the confidence threshold.")
                
                fig2 = plot_confidence_scores(confidence_scores)
                st.pyplot(fig2)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please make sure the file format is correct and contains valid cryptographic data.")

# Explanation section
st.subheader("How It Works")
with st.expander("See methodology"):
    st.markdown("""
    ### Cryptographic Algorithm Identification Process

    1. **Pattern Analysis**
       - Byte frequency distribution examination
       - Block structure detection
       - Entropy and randomness assessment
       - Repeating pattern identification

    2. **Algorithm Matching**
       - Pattern comparison against known cryptographic signatures
       - Statistical similarity to known algorithms
       - Block size and structure analysis
       
    3. **Confidence Assessment**
       - Calculation of match probability for each algorithm
       - Multiple characteristic evaluation
       - Threshold-based determination of final result
    """)

st.sidebar.info("""
### About
This application uses pattern analysis techniques to identify cryptographic algorithms from data samples.
Developed as a research tool for cryptographic analysis.
""")
