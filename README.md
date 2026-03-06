# Hidden Markov Models for Activity Recognition

A machine learning project that uses **Hidden Markov Models (HMM)** for classifying human activities based on accelerometer and gyroscope sensor data. The system recognizes four distinct activities: **standing**, **walking**, **jumping**, and **still**.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This project implements a **sequence-based Hidden Markov Model** for activity recognition using temporal sensor data. Unlike traditional approaches that treat recordings as independent feature vectors, this implementation properly models the temporal dynamics of activities through sequential observations.

### Key Highlights

- **87.5% overall accuracy** on test data
- **Proper temporal modeling** using HMM sequences
- **Multi-sensor fusion** (accelerometer + gyroscope)
- **Rich feature extraction** (90 features per window)
- **Real-world applicability** for activity monitoring systems

## Project Structure

```
Formative2-Hidden-Markov-Models/
│
├── hmm_sequence_model.ipynb          # Main notebook with HMM implementation
├── sequence_segmentation.py          # Segments recordings into sequential windows
├── sequence_feature_extraction.py    # Extracts time & frequency domain features
│
├── recordings/                       # Raw sensor data
│   ├── jumping/                      # Jumping activity recordings
│   ├── standing/                     # Standing activity recordings
│   ├── still/                        # Still activity recordings
│   └── walking/                      # Walking activity recordings
│
├── sequence_data/                    # Segmented window sequences
│   ├── train/                        # Training sequences
│   ├── test/                         # Test sequences
│   └── summary.json                  # Dataset statistics
│
├── sequence_features/                # Extracted feature sequences
│   ├── train/                        # Training feature matrices
│   ├── test/                         # Test feature matrices
│   └── feature_summary.json          # Feature statistics
│
└── hmm_sequence_results/             # Trained models and results
    ├── hmm_jumping.pkl               # Trained jumping HMM
    ├── hmm_standing.pkl              # Trained standing HMM
    ├── hmm_still.pkl                 # Trained still HMM
    ├── hmm_walking.pkl               # Trained walking HMM
    ├── normalization.pkl             # Feature normalization parameters
    ├── evaluation_report.txt         # Performance metrics
    └── evaluation_report.json        # Performance metrics (JSON)
```

##  Features

### 1. **Sequence Segmentation**
- Segments recordings into overlapping time windows (2-second windows, 50% overlap)
- Preserves temporal order for sequence modeling
- Handles variable-length recordings

### 2. **Feature Extraction**
Extracts **90 features** per window:

**Time-Domain Features (45 features):**
- Mean, standard deviation, min, max, range
- Zero-crossing rate, root mean square (RMS)
- Correlation between axes
- Per-axis and magnitude features

**Frequency-Domain Features (45 features):**
- FFT-based spectral features
- Dominant frequency, spectral energy
- Frequency band powers
- Spectral entropy

### 3. **HMM Classification**
- **One HMM per activity class** (4 HMMs total)
- **3 hidden states** per HMM (capturing activity phases)
- **Baum-Welch algorithm** for training (EM-based)
- **Viterbi algorithm** for sequence decoding
- **Log-likelihood scoring** for classification

##  Requirements

- Python 3.7+
- NumPy
- SciPy
- hmmlearn
- scikit-learn
- matplotlib
- seaborn
- pandas

##  Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Formative2-Hidden-Markov-Models
```

2. **Install dependencies:**
```bash
pip install numpy scipy hmmlearn scikit-learn matplotlib seaborn pandas
```

Or using requirements.txt (if available):
```bash
pip install -r requirements.txt
```

##  Usage

### Step 1: Segment Raw Recordings into Sequences
```bash
python sequence_segmentation.py
```
This creates sequential time windows from raw sensor recordings.

### Step 2: Extract Features from Sequences
```bash
python sequence_feature_extraction.py
```
This extracts time-domain and frequency-domain features for each window.

### Step 3: Train and Evaluate HMM Models
Open and run the Jupyter notebook:
```bash
jupyter notebook hmm_sequence_model.ipynb
```

The notebook will:
- Load feature sequences
- Normalize features (z-score normalization)
- Train one HMM per activity class
- Evaluate on test sequences
- Generate confusion matrix and performance metrics
- Save trained models

### Using Trained Models for Prediction
```python
import pickle
import numpy as np

# Load trained models
with open('hmm_sequence_results/hmm_walking.pkl', 'rb') as f:
    hmm_walking = pickle.load(f)

# Load normalization parameters
with open('hmm_sequence_results/normalization.pkl', 'rb') as f:
    norm_params = pickle.load(f)

# Predict on new sequence (normalized feature matrix)
log_likelihood = hmm_walking.score(new_sequence)
```

##  Model Architecture

### Hidden Markov Model Components

1. **Hidden States (Z):** 3 internal states per activity
   - Capture different phases of each activity
   - Example: Walking might have states for left-foot, right-foot, transition

2. **Observations (X):** 90-dimensional feature vectors
   - Time-domain + frequency-domain features
   - Extracted from 2-second windows

3. **Transition Matrix (A):** State transition probabilities
   - Learned via Baum-Welch algorithm
   - Models temporal dynamics within activities

4. **Emission Probabilities (B):** Gaussian distributions
   - One Gaussian per state
   - Models feature distributions in each state

5. **Initial State Probabilities (π):** Starting distribution
   - Learned from training sequences

### Classification Pipeline

```
Raw Recording → Segmentation → Feature Extraction → Normalization 
                     ↓                ↓                   ↓
              [windows]        [feature matrix]      [normalized]
                                      ↓
                              Score with 4 HMMs
                                      ↓
                          Choose highest log-likelihood
                                      ↓
                              Predicted Activity
```

##  Results

### Performance Metrics

| Activity  | Samples | Sensitivity | Specificity | Accuracy |
|-----------|---------|-------------|-------------|----------|
| Jumping   | 4       | 100.0%      | 91.7%       | 100.0%   |
| Standing  | 4       | 75.0%       | 100.0%      | 75.0%    |
| Still     | 4       | 75.0%       | 100.0%      | 75.0%    |
| Walking   | 4       | 100.0%      | 91.7%       | 100.0%   |

**Overall Accuracy: 87.50%**

### Confusion Matrix

```
              jumping  standing  still  walking
jumping          4        0       0       0
standing         1        3       0       0
still            0        0       3       1
walking          0        0       0       4
```

### Key Insights

- **Jumping** and **Walking** are perfectly classified (100% accuracy)
- **Standing** is sometimes confused with jumping (1 misclassification)
- **Still** is sometimes confused with walking (1 misclassification)
- The model effectively captures temporal patterns in dynamic activities

##  Dataset

### Data Collection
- **Sensors:** Accelerometer + Gyroscope
- **Sampling Rate:** Variable (typically 50-100 Hz)
- **Activities:** 4 classes (jumping, standing, still, walking)
- **Recording Duration:** ~10 seconds per recording

### Dataset Split
- **Training:** 34 sequences
- **Testing:** 16 sequences

### Activity Distribution
- **Jumping:** 12 recordings (8 train, 4 test)
- **Standing:** 12 recordings (8 train, 4 test)
- **Still:** 12 recordings (8 train, 4 test)
- **Walking:** 14 recordings (10 train, 4 test)

### Data Format
Each recording contains:
- `Accelerometer.csv`: x, y, z acceleration values with timestamps
- `Gyroscope.csv`: x, y, z angular velocity values with timestamps

##  Technical Details

### Why HMM for Activity Recognition?

1. **Temporal Modeling:** HMMs naturally model sequential patterns
2. **Probabilistic Framework:** Handles sensor noise and variability
3. **Interpretable States:** Hidden states represent activity phases
4. **Efficient Training:** EM algorithm works well with limited data
5. **Real-time Capable:** Fast inference with Viterbi algorithm

### Advantages Over Traditional ML

- **Sequence-aware:** Models temporal dependencies
- **Phase detection:** Captures internal structure of activities
- **Robust:** Handles variable-length sequences
- **Probabilistic:** Provides confidence scores

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is part of an academic formative assessment for Machine Learning coursework at ALU (African Leadership University).

##  Contact

For questions or feedback, please contact the project maintainer.

---

**Project Date:** March 2026  
**Course:** Machine Learning - Formative Assessment  
**Institution:** African Leadership University (ALU)
