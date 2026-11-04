# EEG-Based Emotion Recognition Using Frequency Features and Deep Learning Models

**Author:** Amir Rajabi Heshin  
**Project:** Undergraduate thesis on EEG-based emotion recognition  
**Supervisor:** [Dr. Hossein Aghababa](https://www.loyola.edu/academics/engineering/faculty-staff/aghababa-hossein.html)

## Overview

This project focuses on emotion recognition using electroencephalography (EEG) signals. It began as an undergraduate thesis investigating EEG preprocessing, frequency-domain feature extraction, and deep learning models for emotion classification.

The repository now includes subsequent experiments with classical machine learning baselines, graph neural networks, pretrained EEG foundation models, and Mamba-based architectures. The work covers feature and architecture comparisons, reproduction studies, subject-disjoint validation, and analysis of prediction errors and computational cost. The current experimental focus is **binary emotion classification: negative versus positive** on FACED.

## Dataset

Dataset: *A Large Finer-grained Affective Computing EEG Dataset* ([FACED](https://doi.org/10.1038/s41597-023-02650-w)).

- **Participants:** 123
- **Video clips:** 28 per participant
- **Emotion categories:** 9 — anger, fear, disgust, sadness, neutral, amusement, inspiration, joy, tenderness
- **Preprocessed trials:** last 30 seconds of each video at 250 Hz
- **Binary task:** negative vs. positive; neutral trials excluded

Resources: [Dataset & code](https://www.synapse.org/Synapse:syn50614194) · [Paper & supplement](https://www.nature.com/articles/s41597-023-02650-w).

## Signal Processing and Neural Models

The initial experiments were conducted as part of the undergraduate thesis.

- **Exploration:** raw EEG classification was examined for eight non-neutral emotions and for positive/negative classification. Reported performance was near chance (12.5% and approximately 50%, respectively). The initial pipeline used non-overlapping two-second windows.
- **Feature processing:** DE and PSD were extracted across frequency bands, with filtering, baseline correction, and normalization explored through **14 data-loading configurations**.
- **Model comparison:** Vanilla LSTM, CNN-LSTM, bidirectional LSTM, attention LSTM, multi-branch CNN, Simple and Advanced [EEGNet](https://doi.org/10.1088/1741-2552/aace8c), [DeepConvNet](https://doi.org/10.1002/hbm.23730), and a residual 1D CNN were compared using loading configuration **#8**. This configuration was selected using CNN-LSTM and remained the preferred configuration in the loading-configuration checks.
- **Refinement:** the residual 1D CNN, DeepConvNet, and Simple EEGNet were shortlisted. Changes included dropout, SiLU activations, filter sizes, and classifier regularization. **Six training configurations** varied learning rate, weight decay, batch size, and the binary decision threshold.
- **Error analysis:** precision, recall, confusion matrices, ROC/PR curves, and relationships with arousal, valence, familiarity, and liking were examined. The initial analysis reported greater difficulty for low-arousal/low-liking samples.

### Selected residual CNN

**1D CNN with four residual blocks**, adaptive pooling, and an MLP classifier; loading configuration **#8**, training configuration **#3**. Early stopping ended at epoch **28**, with approximately **43 seconds per epoch**.

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate / weight decay | 0.001 / 0.0001 |
| Batch size / decision threshold | 64 / 0.5 |

| Reported metric | Training | Test |
| --- | ---: | ---: |
| Accuracy | 89.15% | **72.14%** |
| Loss | 0.3933 | 0.6482 |

**Folders:** `models`, `configs`, `Analyze`, `results`, `utils`.
