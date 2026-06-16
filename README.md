# Multi-Class Imagined Speech Classification using EEG Signals

## Overview

This graduation project investigates the classification of imagined speech from EEG signals using deep learning techniques. The work is based on the **BCI Competition 2020 – Track 3 Imagined Speech Dataset**, where participants imagine speaking specific words while their brain activity is recorded through a 64-channel EEG system.

While the original dataset contains five imagined speech classes, this project focuses on a subset of three classes:

* Hello
* Help Me
* Stop

The objective is to develop a robust Brain-Computer Interface (BCI) capable of recognizing imagined speech directly from EEG recordings.

---

## Dataset

The dataset was collected from **15 healthy subjects (20–30 years old)** and released as part of the **International BCI Competition 2020**.

### Original Classes

1. Hello
2. Help Me
3. Stop
4. Thank You
5. Yes

### Classes Used in This Project

To simplify the classification task and focus on high-quality discrimination, only the following classes were retained:

* Hello
* Help Me
* Stop

### Dataset Structure

For each subject:

| Split      | Trials       |
| ---------- | ------------ |
| Training   | 300 labeled  |
| Validation | 50 labeled   |
| Test       | 50 unlabeled |

Each subject's data is stored in MATLAB format (`.mat`) files.

---

# Notebook Workflow

The notebook follows the complete EEG processing and classification pipeline shown below:

```text
Raw EEG Signals (.mat)
           │
           ▼
Data Loading & Parsing
           │
           ▼
Channel Selection
           │
           ▼
Signal Preprocessing
   ├─ 40 Hz Notch Filter
   └─ Common Average Referencing (CAR)
           │
           ▼
Temporal Windowing
(125 ms windows, 50% overlap)
           │
           ▼
EEG Topographic Mapping
           │
           ▼
RGB Spatial Brain Maps
           │
           ▼
3-Class Filtering
(Hello, Help Me, Stop)
           │
           ▼
Train / Validation / Test Split
           │
           ▼
Data Augmentation
(Gaussian Noise Injection)
           │
           ▼
3D CNN Training
(Keras Tuner Optimization)
           │
           ▼
9-Fold Cross Validation
           │
           ▼
Performance Evaluation
   ├─ Accuracy
   ├─ Precision
   ├─ Recall
   ├─ F1 Score
   └─ Confusion Matrix
```

---

## Signal Preprocessing

Several preprocessing stages were applied before model training:

### 1. Notch Filtering

A digital notch filter centered at **40 Hz** was applied to suppress power-line and environmental interference while preserving useful neural activity.

### 2. Common Average Referencing (CAR)

Common Average Referencing was used to reduce global noise and improve the signal-to-noise ratio across EEG channels.

### 3. Temporal Windowing

The EEG signals were segmented into:

* Window Length: **125 ms**
* Overlap: **50%**
* Sampling Frequency: **256 Hz**

This produces multiple temporal snapshots that preserve both spatial and temporal information.

---

## EEG Topographic Representation

Instead of directly feeding raw EEG signals into the network, electrode values are projected onto a 2D scalp layout to generate topographic brain activity maps.

The resulting representations:

* Preserve spatial relationships between electrodes.
* Convert EEG signals into image-like structures.
* Enable the use of convolutional neural networks.

Each temporal window is transformed into an RGB topographic image, creating a sequence of brain activity maps over time.

---

## Model Architecture

The proposed model is based on a **3D Convolutional Neural Network (3D-CNN)** designed to learn both spatial and temporal EEG patterns simultaneously.

### Architecture Overview

```text
Input Topographic Maps
        │
        ▼
3D Convolution Layer
        │
        ▼
Batch Normalization
        │
        ▼
Max Pooling
        │
        ▼
3D Convolution Layer
        │
        ▼
Batch Normalization
        │
        ▼
Max Pooling
        │
        ▼
Flatten
        │
        ▼
Dense Layer
        │
        ▼
Dropout
        │
        ▼
Softmax Output
(3 Classes)
```

### Hyperparameter Optimization

The architecture was automatically optimized using **Keras Tuner**, including:

* Number of convolution filters
* Kernel sizes
* Dense layer size
* Dropout rate
* Learning rate

---

## Data Augmentation

To improve generalization and reduce overfitting, targeted data augmentation was performed by injecting low-amplitude Gaussian noise into EEG samples.

This approach:

* Increases training diversity.
* Preserves class characteristics.
* Improves model robustness.

---

## Evaluation

Model performance is assessed using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

A **9-Fold Cross-Validation** strategy is employed to ensure reliable performance estimation across subjects and trials.

---

## Technologies Used

* Python
* TensorFlow / Keras
* Keras Tuner
* NumPy
* SciPy
* Pandas
* Scikit-Learn
* Matplotlib

---

## Acknowledgments

This project uses data from the **2020 International BCI Competition (Track 3: Imagined Speech Classification)**.

Dataset Source:
https://osf.io/pq7vb/

All credit for data collection, experimental design, and dataset publication belongs to the original authors and competition organizers.
