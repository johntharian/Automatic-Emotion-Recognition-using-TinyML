# 🎭 AERSUT – Automatic Emotion Recognition System using TinyML  (IEEE research publication)

### Understanding Emotions, One Microcontroller at a Time  

---

## 🌟 About the IEEE Project  

AERSUT (**Automatic Emotion Recognition System using TinyML**) is a research project that brings the power of **deep learning** and **speech-based emotion recognition** into the world of **TinyML**.  

The core idea:  
> Can a tiny device with very limited memory and compute power actually **understand how we feel** just from the way we speak?  

Turns out — yes, it can.  

---

## 🎤 How It Works  

AERSUT takes in **raw speech audio** and passes it through three stages:  

1. **Input & Preprocessing**  
   - Speech is recorded from different speakers.  
   - Audio is cleaned, stretched, shifted, and even injected with noise for robustness.  

2. **Feature Extraction**  
   - Key features like **MFCCs**, **Mel-Spectrograms**, **Zero-Crossing Rate**, and **RMS Energy** are pulled from the audio.  
   - These features capture the unique patterns of human emotion in sound.  

3. **Emotion Classification**  
   - Deep learning models (CNN & CNN-LSTM) trained on **RAVDESS** and **CREMA-D** datasets classify the speech into **8 emotions**:  
     😮 Surprise | 😐 Neutral | 🤢 Disgust | 😨 Fear | 😢 Sad | 😌 Calm | 😀 Happy | 😡 Anger  

The trained model is then compressed into **TensorFlow Lite** and deployed on a **TinyML microcontroller board**.  

---

## 📊 Results  

| Model      | Training Accuracy | Test Accuracy |
|------------|------------------|---------------|
| **CNN**    | 99%              | 67%           |
| **CNN-LSTM** | 96%              | 72%           |

- **CNN** learned patterns quickly but struggled with generalization.  
- **CNN-LSTM** balanced feature extraction with sequence understanding, leading to **better test accuracy**.  
- This shows that while deep CNNs are powerful, **adding temporal context (LSTM)** makes the system more reliable for real-world speech.  

---

## 🔗 How the Parts Connect  

- **Datasets (RAVDESS + CREMA-D):** Provide thousands of labeled emotional audio clips.  
- **Feature Extraction:** Converts raw waveforms into meaningful numbers computers can understand.  
- **Deep Learning Models:** Learn correlations between audio features and emotions.  
- **TinyML Deployment:** Shrinks the model into a size that fits on a microcontroller, allowing **real-time, on-device emotion detection** without the cloud.  

---

## 🚀 Why It Matters  

- **Human-Computer Interaction** → Devices that don’t just listen, but also understand.  
- **Healthcare & Well-being** → Tools that can detect stress, depression, or emotional states from voice.  
- **Education & Customer Service** → Systems that adapt based on learner or customer emotions.  
- **Edge AI** → Privacy-friendly since data never leaves the device.  

---

## 🌱 Future Directions  

- Expand emotion recognition across **multiple languages**.  
- Enhance accuracy with **hybrid models** and **attention layers**.  
- Integrate with **IoT devices** and **wearables** for real-world applications.  
- Move towards **real-time emotion-aware assistants**.  

---

## 🧾 Citation  

If you use this work, please cite:  
