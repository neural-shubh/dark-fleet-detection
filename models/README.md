# Models

The trained model files are hosted on Hugging Face due to GitHub's file size limits.

## Download from Hugging Face

🤗 **[neural-shubh/dark-fleet-models](https://huggingface.co/neural-shubh/dark-fleet-models)**

### Files

| File | Branch | Size | Description |
|------|--------|------|-------------|
| `cnn_branch.keras` | CNN | ~14 MB | MobileNetV2 SAR vessel density classifier |
| `lstm_best.keras` | LSTM | ~2 MB | Bidirectional LSTM dark vessel detector |
| `rnn_best.keras` | RNN | ~1 MB | SimpleRNN positional anomaly detector |
| `lstm_scaler.pkl` | LSTM/RNN | ~1 KB | StandardScaler fitted on training features |
| `fusion_meta_model.pkl` | Fusion | ~1 KB | Logistic regression meta-learner |

### Download via Python

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf
import joblib

# CNN branch
cnn_path = hf_hub_download(
    repo_id="neural-shubh/dark-fleet-models",
    filename="cnn_branch.keras"
)
cnn_model = tf.keras.models.load_model(cnn_path)

# LSTM branch
lstm_path = hf_hub_download(
    repo_id="neural-shubh/dark-fleet-models",
    filename="lstm_best.keras"
)
lstm_model = tf.keras.models.load_model(lstm_path)

# RNN branch
rnn_path = hf_hub_download(
    repo_id="neural-shubh/dark-fleet-models",
    filename="rnn_best.keras"
)
rnn_model = tf.keras.models.load_model(rnn_path)

# Scaler + Fusion
scaler = joblib.load(hf_hub_download(
    repo_id="neural-shubh/dark-fleet-models",
    filename="lstm_scaler.pkl"
))
fusion_model = joblib.load(hf_hub_download(
    repo_id="neural-shubh/dark-fleet-models",
    filename="fusion_meta_model.pkl"
))

print("All models loaded ✅")
```

### Or download manually

1. Go to [huggingface.co/neural-shubh/dark-fleet-models](https://huggingface.co/neural-shubh/dark-fleet-models)
2. Click each file and hit **Download**
3. Place all files in this `models/` folder

### Install huggingface_hub

```bash
pip install huggingface_hub
```

---

## Model Performance

| Branch | AUC | Accuracy |
|--------|-----|----------|
| CNN | 0.941 | 87.4% |
| LSTM | 0.911 | 85.1% |
| RNN | 0.927 | 86.0% |
| Fusion | 0.926 | 88.0% |

---

## Quick Inference Example

```python
import numpy as np

# Your vessel sequence — shape (1, 2, 12)
# Features: lat, lon, presence_score, length_m, fishing_score,
#           matching_score, hour, dayofweek,
#           lat_diff, lon_diff, speed_proxy, impossible_speed

sequence = np.array([[
    [12.5, 72.3, 0.99, 180.0, 0.02, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    [12.8, 74.1, 0.97, 180.0, 0.02, 0.0, 2.1, 5.0, 0.3, 1.8, 0.002, 1.0]
]], dtype=np.float32)

# Scale
sequence_scaled = scaler.transform(
    sequence.reshape(-1, 12)
).reshape(1, 2, 12)

# Run inference
lstm_score   = lstm_model.predict(sequence_scaled, verbose=0)[0][0]
rnn_score    = rnn_model.predict(sequence_scaled, verbose=0)[0][0]
fusion_score = fusion_model.predict_proba([[lstm_score, rnn_score]])[0][1]

print(f"LSTM score:   {lstm_score:.4f}")
print(f"RNN score:    {rnn_score:.4f}")
print(f"Fusion score: {fusion_score:.4f}")
print(f"Verdict:      {'DARK VESSEL ⚠️' if fusion_score > 0.5 else 'Normal ✅'}")
```
