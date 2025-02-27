# Fourier Decomposition and Autoencoder-based EEG Classification

## Overview
This project implements an EEG signal classification pipeline using Fourier Decomposition Method (FDM) and an Autoencoder-based feature extraction approach. The extracted features are used for training a classifier to predict mental states.

## Features
- **Fourier Decomposition Method (FDM)**: Decomposes EEG signals into frequency bands.
- **Feature Extraction**: Computes statistical features like mean, variance, skewness, kurtosis, and entropy.
- **Autoencoder Model**: Learns a compressed representation of the EEG features.
- **Classification Model**: Uses the encoded features for classification.
- **Class Weight Balancing**: Adjusts for imbalanced data using `compute_class_weight`.

## Dataset
The input dataset should be a CSV file containing EEG features with a label column. By default, the label column is named `Label`.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- SciPy
- Scikit-learn
- TensorFlow/Keras

## File Structure
- `mental-state.csv` - EEG dataset 
- `main.py` - The main script for training the autoencoder and classifier.

## How to Run
1. Install dependencies using:
   ```bash
   pip install numpy pandas scipy scikit-learn tensorflow
   ```
2. Place the EEG dataset (`mental-state.csv`) in the working directory.
3. Run the script:
   ```bash
   python main.py
   ```

## Code Explanation
### Fourier Decomposition Method (FDM)
Decomposes signals into different frequency bands using Discrete Cosine Transform (DCT) and Inverse DCT.

### Feature Extraction
Extracts the following statistical features from the EEG signals:
- Mean amplitude
- Variance
- Skewness
- Kurtosis
- Signal entropy

### Autoencoder Model
A feedforward neural network with an encoder-decoder architecture to learn latent representations.

### Classification Model
Uses the encoded features to train a classifier with a softmax activation function.

## Expected Output
- Training and validation accuracy for the classifier.
- Classification report showing precision, recall, and F1-score.
- Final model accuracy.

## Contact
For any issues or improvements, please open an issue in the repository or contact the project maintainer.

