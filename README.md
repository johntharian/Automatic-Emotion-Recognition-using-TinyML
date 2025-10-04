# Automatic Speech Emotion Recognition using TinyML (AERSUT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This paper proposes AERSUT (Automatic Emotion Recognition System Using TinyML), a system that detects and analyzes emotions from speech signals using TinyML. The system classifies emotions into eight categories: surprise, neutral, disgust, fear, sad, calm, happy, and anger. The implementation focuses on running efficiently on resource-constrained devices while maintaining high accuracy.

## Key Features

- **Multi-emotion Classification**: Detects 8 distinct emotional states from speech
- **Advanced Feature Extraction**: Utilizes MFCCs, Mel-spectrograms, Zero Crossing Rate, and RMS Energy
- **Data Augmentation**: Implements noise addition, time stretching, and signal shifting for robust training
- **TinyML Integration**: Optimized for deployment on resource-constrained devices
- **Dual-Model Approach**: Implements both CNN and CNN-LSTM architectures for performance comparison
- **High Accuracy**: Achieves up to 72% test accuracy on the combined dataset

## Dataset

The system uses a combination of two benchmark datasets, totaling approximately 20,000 audio samples:

### 1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Size**: 1,440 speech samples
- **Actors**: 24 professional actors (12 male, 12 female)
- **Emotions**: 8 emotional states (neutral, calm, happy, sad, angry, fear, disgust, surprise)
- **Format**: 16-bit, 48kHz .wav files

### 2. CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Size**: 7,442 audio clips
- **Actors**: 91 actors (48 male, 43 female)
- **Diversity**: Ages 20-74, various races and ethnicities
- **Emotions**: 6 emotional states (angry, disgust, fear, happy, neutral, sad)

### Data Augmentation
To enhance model robustness, the following augmentation techniques were applied:
- **Noise Injection**: Adding Gaussian noise to audio signals
- **Time Stretching**: ±20% variation in speech rate
- **Pitch Shifting**: Modulating pitch by ±3 semitones
- **Time Shifting**: Randomly shifting audio in time

## System Architecture

### 1. Feature Extraction Pipeline

#### A. Mel Spectrograms
- Visual representation of speech's temporal and spectral changes
- Captures rich emotional features not present in text/transcripts
- Provides 2D feature maps for CNN processing

#### B. Mel Frequency Cepstral Coefficients (MFCCs)
- 13 coefficients extracted per frame
- Pre-emphasis to enhance high frequencies
- Mel-scale transformation for human-like frequency perception
- DCT for decorrelation of filter bank energies

#### C. Additional Features
- **Zero Crossing Rate**: Measures signal noisiness and periodicity
- **Root Mean Squared Energy**: Represents signal power over time

### 2. Model Architectures

#### A. CNN Model
- **Input**: MFCC features (13 coefficients × 130 frames)
- **Convolutional Blocks**:
  - Conv1D (256 filters, kernel=5) → BatchNorm → ReLU → MaxPool
  - Conv1D (128 filters, kernel=3) → BatchNorm → ReLU → MaxPool
  - Conv1D (64 filters, kernel=3) → BatchNorm → ReLU → MaxPool
- **Classification Head**:
  - Flatten → Dense(1024, ReLU) → Dropout(0.5)
  - Dense(8, softmax)

#### B. CNN-LSTM Model
- **Feature Extraction**: CNN layers similar to above
- **Temporal Modeling**: LSTM layers to capture sequential dependencies
- **Training**: 120 epochs with learning rate 0.00001
- **Performance**: 96% training accuracy, 72% test accuracy

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Librosa
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
- IPython
- SoundFile

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Automatic-Emotion-Recognition-using-TinyML.git
   cd Automatic-Emotion-Recognition-using-TinyML
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**
   - Download the RAVDESS and CREMA-D datasets
   - Place them in the `input/` directory with the following structure:
     ```
     input/
     ├── ravdess-emotional-speech-audio/
     │   └── audio_speech_actors_01-24/
     └── cremad/
         └── AudioWAV/
     ```

2. **Run the Emotion Recognition Pipeline**
   ```bash
   python emotion_recognition.py
   ```

## Data Augmentation

The system includes several data augmentation techniques to improve model generalization:
- **Noise Addition**: Adds random Gaussian noise to audio signals
- **Time Stretching**: Slows down or speeds up the audio
- **Pitch Shifting**: Modifies the pitch of the audio
- **Time Shifting**: Shifts the audio in time

## Feature Extraction

- **MFCCs (Mel-frequency cepstral coefficients)**: 13 coefficients
- **Delta and Delta-Delta**: First and second derivatives of MFCCs
- **Feature Normalization**: Standard scaling of features

## Training & Evaluation

### Training Configuration
- **Framework**: TensorFlow/Keras
- **Environment**: Google Colab with GPU acceleration
- **Epochs**: 120
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate=0.00001)
- **Loss Function**: Categorical Cross-Entropy
- **Callbacks**:
  - Learning Rate Reduction on Plateau
  - Model Checkpointing
  - Early Stopping

### Performance Comparison

| Model       | Training Accuracy | Test Accuracy |
|-------------|-------------------|---------------|
| CNN         | 99%               | 67%           |
| CNN-LSTM    | 96%               | 72%           |

### Key Findings
- CNN-LSTM showed better generalization (higher test accuracy)
- CNN showed signs of overfitting (99% training vs 67% test)
- The combined use of MFCCs and mel-spectrograms provided robust features
- Data augmentation significantly improved model robustness

## Evaluation

Model performance is evaluated using:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## TinyML Deployment

The trained models were optimized for edge deployment using TensorFlow Lite:

### Conversion to TensorFlow Lite
```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('emotion_recognition.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Deployment on Edge Devices
- **Hardware**: Compatible with various TinyML boards
- **Inference**: Real-time emotion classification
- **Input**: Audio from onboard microphone
- **Output**: Predicted emotion class with confidence score

## Visualization & Analysis

### 1. Audio Analysis
- **Waveform Visualization**: Time-domain representation of audio signals
- **Spectrograms**: Frequency content over time for different emotions
- **MFCC Plots**: Visualizing cepstral coefficients

### 2. Model Performance
- **Learning Curves**: Training/validation accuracy and loss over epochs
- **Confusion Matrices**: Per-class performance analysis
- **Feature Importance**: Analysis of which features contribute most to classification

### 3. Real-time Monitoring
- Live visualization of input audio features
- Real-time emotion classification feedback

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RAVDESS Dataset: https://zenodo.org/record/1188976
- CREMA-D Dataset: https://github.com/CheyneyComputerScience/CREMA-D
