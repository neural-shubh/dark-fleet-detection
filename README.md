# 🛸 Dark Fleet Detection — CNN + LSTM + RNN Fusion

> A multimodal deep learning framework for detecting dark fleet vessels in the Indian Ocean using Sentinel-1 SAR imagery and Global Fishing Watch vessel detection data.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Models-neural--shubh-yellow)](https://huggingface.co/neural-shubh/dark-fleet-models)

---

## 📌 Overview

Dark fleet vessels — ships that disable or manipulate their AIS transponders to evade detection — have surged since 2022, enabling sanctioned states to route crude oil and strategic cargo through the Indian Ocean undetected.

This project proposes a **three-branch deep learning fusion system** that combines:

- 🛰️ **CNN branch** — classifies SAR image patches by vessel density (MobileNetV2 transfer learning)
- 🚢 **LSTM branch** — detects dark vessel behavioural patterns from vessel movement sequences (Bidirectional LSTM)
- ⚠️ **RNN branch** — flags abrupt positional and confidence anomalies between consecutive detections (SimpleRNN)
- 🔗 **Fusion layer** — logistic regression meta-learner combining LSTM + RNN scores

Trained on open-access satellite data, filtered to the **Indian Ocean Region**, and evaluated against held-out test sets.

---

## 📊 Results

| Branch | AUC | Accuracy | Precision | Recall |
|--------|-----|----------|-----------|--------|
| CNN (SAR density) | 0.941 | 87.4% | 91.0% | 81.4% |
| LSTM (trajectory) | 0.911 | 85.1% | 68.1% | 80.0% |
| RNN (anomaly) | 0.927 | 86.0% | 69.8% | 82.2% |
| **Fusion (LSTM+RNN)** | **0.926** | **88.0%** | **78.0%** | **72.0%** |

> 66.3% of SAR-detected vessels in the Indian Ocean were unmatched to AIS records — confirming a substantial dark fleet surveillance gap.

---

## 🗂️ Dataset

| Dataset | Branch | Source |
|---------|--------|--------|
| SARscope (HRSID + OPEN-SSDD) | CNN | [Kaggle / Roboflow](https://universe.roboflow.com) |
| GFW Sentinel-1 SAR Vessel Detections | LSTM, RNN | [Global Fishing Watch](https://globalfishingwatch.org) |

**Indian Ocean filter:** lat −30° to 30°N, lon 40° to 100°E  
**GFW records:** 107,257 total → 16,795 after filtering → 11,136 unmatched (66.3%)

---

## 🏗️ Architecture

```
SAR Imagery ──► CNN Branch (MobileNetV2) ──► Spatial Triage Score
                                                      │
GFW Vessel    ──► LSTM Branch (BiLSTM)  ──► Dark Score ──► Logistic Regression ──► Final Risk Score
Sequences     ──► RNN Branch (SimpleRNN)──► Dark Score ──►        (Fusion)
```

**CNN Branch**
- MobileNetV2 backbone (ImageNet pretrained)
- Two-phase training: frozen base → fine-tune last 30 layers
- Input: 96×96 SAR image patches
- Label: high density (≥2 vessels) vs low density

**LSTM Branch**
- Bidirectional LSTM (64→32 units)
- Input: 2-detection vessel sequences, 12 features
- Features: lat, lon, presence_score, length_m, fishing_score, matching_score, hour, dayofweek, lat_diff, lon_diff, speed_proxy, impossible_speed

**RNN Branch**
- SimpleRNN (64→32 units, tanh activation)
- Same input format as LSTM
- Sensitive to step-to-step transitions in confidence and positional features

**Fusion Layer**
- Logistic regression meta-learner on stacked LSTM + RNN scores
- Learned weights: LSTM 4.699, RNN 3.905

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/neural-shubh/dark-fleet-detection.git
cd dark-fleet-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
- SARscope: Download from Kaggle/Roboflow and place in `data/sarscope/`
- GFW data: Download from [globalfishingwatch.org](https://globalfishingwatch.org/data-download/datasets/public-sentinel-1-detections:v20231026) and place in `data/gfw/`

### 4. Run the notebook
```
notebooks/
└── dark_fleet_detection_pipeline.ipynb   # Complete pipeline — CNN, LSTM, RNN, Fusion
```

### 5. Load pretrained models
```python
import tensorflow as tf
import joblib
import numpy as np

# CNN branch
cnn_model = tf.keras.models.load_model('models/cnn_branch.keras')

# LSTM branch
lstm_model = tf.keras.models.load_model('models/lstm_best.keras')
scaler = joblib.load('models/lstm_scaler.pkl')

# RNN branch
rnn_model = tf.keras.models.load_model('models/rnn_best.keras')

# Fusion
fusion_model = joblib.load('models/fusion_meta_model.pkl')

# Run inference
lstm_score = lstm_model.predict(sequence_scaled)[0][0]
rnn_score  = rnn_model.predict(sequence_scaled)[0][0]
risk_score = fusion_model.predict_proba([[lstm_score, rnn_score]])[0][1]
print(f"Dark fleet risk score: {risk_score:.4f}")
```

---

## 📁 Repository Structure

```
dark-fleet-detection/
├── notebooks/
│   └── dark_fleet_detection_pipeline.ipynb   # Complete pipeline
├── models/                    # Download from Hugging Face
│   └── README.md              # Instructions to download
├── data/
│   └── README.md              # Dataset download instructions
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** Trained model files are hosted on Hugging Face at [neural-shubh/dark-fleet-models](https://huggingface.co/neural-shubh/dark-fleet-models) due to file size limits.

---

## 📄 Paper

This project accompanies the research paper:

**"Monitoring Dark Fleets and Maritime Domain Awareness: A Multimodal CNN-LSTM-RNN Framework for the Indian Ocean Region"**

> Available on Zenodo: https://doi.org/10.5281/zenodo.21422532

---

## 🔑 Key Findings

- **66.3%** of Indian Ocean SAR vessel detections in March 2026 were unmatched to AIS — far above the global average of ~33%
- All three branches exceed **AUC 0.91** on held-out test data
- Fusion model achieves **88% accuracy** and **78% dark vessel precision**
- The framework is viable for operational triage within maritime surveillance workflows such as India's IFC-IOR

---

## ⚠️ Limitations

- Labels are derived from AIS matching status, not independently verified dark fleet behaviour
- Sophisticated spoofing (AIS transmitting but position falsified) would evade detection
- GFW data provides monthly snapshots — most vessels appear only once, limiting sequence length
- Intended as analyst support tool, not autonomous enforcement

---

## 🛠️ Requirements

```
tensorflow>=2.12
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
pycocotools
joblib
Pillow
matplotlib
seaborn
```

---

## 📜 License

MIT License — free to use for research and educational purposes.

---

## 🙏 Acknowledgements

- [Global Fishing Watch](https://globalfishingwatch.org) for open-access SAR vessel detection data
- [Paolo et al. (2024)](https://www.nature.com/articles/s41586-023-06825-8) for the foundational dark vessel detection methodology
- [SARscope / Roboflow](https://universe.roboflow.com) for the SAR ship detection dataset
- Google Colab for GPU compute

---

*Built as part of an undergraduate research project on maritime domain awareness and deep learning.*
