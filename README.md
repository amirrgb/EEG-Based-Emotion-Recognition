# EEG-Based Emotion Recognition Using Frequency Features and Deep Learning Models

## Overview

This project focuses on emotion recognition using electroencephalography (EEG) signals. The main objective is to classify emotional states based on EEG recordings using a range of deep learning models. The work includes comprehensive preprocessing of EEG data, feature extraction, deep learning-based classification, and an in-depth analysis of performance and model behavior.

## Dataset

The dataset used in this project is the **FACED** dataset:

* Contains 30-second EEG recordings of subjects watching affective video stimuli.
* Provides both **discrete emotion labels** (8 classes) and **continuous emotional dimensions** (arousal, valence, familiarity, liking).
* Each EEG recording is segmented into 2-second non-overlapping windows for data augmentation.

## Project Steps

### 1. Raw Data Exploration and Baseline Testing

* Conducted experiments directly on raw EEG signals (without feature extraction).
* Attempted classification into 8 emotion classes.
* Observed performance near chance level (12.5%), indicating insufficient model learning.
* Also attempted binary classification (positive vs negative), which again resulted in near-chance accuracy (\~50%).

### 2. Feature Extraction and Signal Processing

* Extracted features: **Power Spectral Density (PSD)** and **Differential Entropy (DE)** across multiple frequency bands.
* Explored different filtering methods and preprocessing strategies (e.g., baseline correction, normalization).
* Constructed 14 different **data load configurations (load configs)** to test different preprocessing pipelines.

### 3. Model Selection

Nine deep learning models were implemented and compared:

* **Vanilla LSTM**
* **CNN + LSTM Hybrid**
* **Bidirectional LSTM**
* **LSTM with Attention Mechanism**
* **Multi-Branch CNN**
* **Simple EEGNet**
* **Advanced EEGNet**
* **DeepConvNet**
* **1D CNN with Residual Blocks**

All models were tested on the same data configuration (#8) to ensure a fair comparison.

### 4. Load Configuration Selection

* Selected **load config #8** using comparative performance on CNN\_LSTM.
* This configuration showed the best performance across preprocessing methods.

### 5. Model Refinement and Training Config Optimization

* Top 3 models were selected based on performance:

  * **1D\_CNN\_Residual\_Blocks**
  * **DeepConvNet**
  * **Simple\_EEGNet**

* Applied architectural improvements:

  * Added dropout layers, changed activation functions to **SiLU**, adjusted filter sizes.
  * Improved classifier sections with hidden layers and regularization.

* Defined 6 training configurations with changes to:

  * **Learning Rate**
  * **Weight Decay**
  * **Batch Size**
  * **Sigmoid Thresholds** (for binary classification)

* Evaluated model robustness and stability across these configs.

* Final selection: **1D\_CNN\_Residual\_Blocks** with config ID **3** (best generalization and performance).

### 6. Final Evaluation and Error Analysis

* Compared model predictions vs true labels.
* Analyzed classification difficulty across:

  * Emotion class
  * Arousal, valence, familiarity, liking scores
* Observed higher difficulty in low-arousal/low-liking samples.
* Used metrics like precision, recall, confusion matrix, ROC/PR curves.

### 7. Generalization Testing

* Evaluated model on various data configurations.
* Best generalization observed with **load config #8** across all training setups.

## Key Results

| Metric               | Value        |
| -------------------- | ------------ |
| Final Test Accuracy  | **72.14%**   |
| Final Train Accuracy | 89.15%       |
| Final Test Loss      | 0.6482       |
| Final Train Loss     | 0.3933       |
| Epochs (Early Stop)  | 28           |
| Avg Epoch Time       | \~43 seconds |

## Final Model

* **Architecture**: 1D CNN with 4 residual blocks, adaptive pooling, MLP classifier.
* **Training Setup**:

  * Optimizer: Adam
  * Learning Rate: 0.001
  * Weight Decay: 0.0001
  * Batch Size: 64
  * Sigmoid Threshold: 0.5

## Future Work Suggestions

1. **Generalization Study**

   * Apply LOSO-CV to evaluate subject-independent generalization.
   * Analyze model per-subject performance.

2. **Neurophysiological Feature Analysis**

   * Use Gradient SHAP, saliency maps, or permutation importance to understand feature contributions.
   * Identify key EEG channels or frequency bands per emotion.

3. **Temporal Dynamics**

   * Use sliding window inference to detect emotion dynamics over time.
   * Analyze whether specific time segments (e.g., initial 2s) are more informative.

4. **Multi-Dataset Validation**

   * Test on other public datasets: DEAP, SEED, DREAMER.

5. **Threshold Tuning**

   * Use ROC/PR curves to find optimal decision threshold for sigmoid outputs.

6. **Metadata Utilization**

   * Integrate gender, age, and video metadata.
   * Incorporate electrode positions for GNN-based architectures.

## Tools & Libraries

* Python 3.x
* PyTorch
* Scikit-learn
* NumPy / Pandas / Matplotlib / Seaborn

## Citation and References

* Lawhern et al., "EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces", JNE 2018
* Schirrmeister et al., "Deep learning with CNNs for EEG decoding", Human Brain Mapping, 2017
* Koelstra et al., "DEAP dataset", IEEE TAC, 2012
* Chen et al., "FACED: A Finer-Grained EEG Dataset for Affective Computing", Sci Data, 2023

---

**Author**: Amir RGB
**Project**: Undergraduate Final Thesis – Emotion Recognition with EEG & Deep Learning
