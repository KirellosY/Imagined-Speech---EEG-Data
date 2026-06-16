# Multi-Class Imagined Speech Classification using EEG Signals

## Overview

This graduation project investigates EEG-based imagined speech recognition using deep learning techniques. The work is based on the BCI Competition 2020 Track 3 dataset and focuses on classifying three imagined speech commands (**Hello**, **Help Me**, and **Stop**) from non-invasive EEG recordings.

The proposed framework combines EEG preprocessing, topographic brain mapping, data augmentation, and a 3D Convolutional Neural Network (3D-CNN) to learn spatial-temporal neural representations associated with imagined speech.

---

# Results

The proposed 3D-CNN model was evaluated on a three-class imagined speech classification task using EEG recordings from the BCI Competition 2020 dataset. The model was trained on topographic EEG representations generated after preprocessing and spatial mapping of neural activity.

## Performance Summary

| Metric | Value |
|----------|----------|
| Accuracy | 63.3% |
| Macro Precision | 63.1% |
| Macro Recall | 63.3% |
| Macro F1-Score | 62.8% |

## Confusion Matrix

<p align="center">
  <img src="figures/confusion_matrix.png" width="650">
</p>

### Analysis

The confusion matrix demonstrates that the model successfully distinguishes between the three imagined speech classes, achieving the highest recognition performance for the **Hello** class.

Key observations:

- **Hello** achieved the strongest classification performance, with 8 out of 10 samples correctly identified.
- **Help Me** showed moderate performance, with most errors occurring when samples were confused with **Hello**.
- **Stop** was the most challenging class, frequently being misclassified as **Help Me**.
- The model demonstrates meaningful discriminative capability despite the inherent difficulty and low signal-to-noise ratio of imagined speech EEG data.

## Precision, Recall and F1-Score per Class

<p align="center">
  <img src="figures/classification_metrics.png" width="700">
</p>

### Class-wise Performance

| Class | Precision | Recall | F1-Score |
|---------|---------|---------|---------|
| Hello | 0.67 | 0.80 | 0.73 |
| Help Me | 0.60 | 0.60 | 0.60 |
| Stop | 0.63 | 0.50 | 0.56 |

### Analysis

The class-wise evaluation highlights variations in classification difficulty:

- **Hello** achieved the highest recall (80%), indicating that the model consistently recognizes this imagined speech category.
- **Help Me** maintained balanced precision and recall values, demonstrating stable classification behavior.
- **Stop** exhibited lower recall, suggesting greater overlap in neural patterns with other imagined speech classes.
- The overall results indicate that the proposed framework successfully captures meaningful spatial-temporal EEG features for imagined speech decoding.

---

## Dataset Description

This project utilizes the **BCI Competition 2020 – Track 3: Imagined Speech Classification Dataset**, a publicly available EEG dataset designed for imagined speech recognition research.

EEG signals were recorded from **15 healthy participants (20–30 years old)** using a **64-channel EEG acquisition system** sampled at **256 Hz**.

Participants were instructed to imagine speaking one of five words or phrases commonly used in daily communication:

- Hello
- Help Me
- Stop
- Thank You
- Yes

For this project, only three classes were selected:

- Hello
- Help Me
- Stop

### Experimental Protocol

Each trial followed a structured imagined-speech paradigm:

1. An auditory cue (target word) was presented for **2 seconds**.
2. A fixation cross appeared for **0.8–1.2 seconds**.
3. The participant imagined pronouncing the instructed word.
4. Steps 2–3 were repeated **four times** for the same cue.
5. A **3-second relaxation period** followed before the next trial.

This design allows multiple imagined speech repetitions to be collected for each target word while minimizing muscular speech activity.

---

# Project Workflow

```text
Raw EEG Signals (.mat files)
            │
            ▼
Data Loading
            │
            ▼
Class Selection
(Hello, Help Me, Stop)
            │
            ▼
EEG Preprocessing
 ├── Notch Filtering
 └── Common Average Referencing (CAR)
            │
            ▼
Signal Segmentation
(Temporal Windowing)
            │
            ▼
EEG Topographic Mapping
            │
            ▼
RGB Brain Activity Images
            │
            ▼
Dataset Balancing & Augmentation
(Gaussian Noise Injection)
            │
            ▼
Train / Validation Split
            │
            ▼
3D CNN Model
            │
            ▼
Hyperparameter Optimization
(Keras Tuner)
            │
            ▼
9-Fold Cross Validation
            │
            ▼
Performance Evaluation
 ├── Accuracy
 ├── Precision
 ├── Recall
 ├── F1 Score
 └── Confusion Matrix
