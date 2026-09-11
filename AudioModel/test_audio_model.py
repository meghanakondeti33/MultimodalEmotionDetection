import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model


# ==========================================
# Paths
# ==========================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "AudioModel",
    "best_model_BiLSTM_2D_CNN.keras"
)

# We need a WAV file here
AUDIO_PATH = os.path.join(
    PROJECT_DIR,
    "AudioModel",
    "test_audio.wav"
)


# ==========================================
# Emotion labels
# ==========================================

labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ==========================================
# Check files
# ==========================================

print("Model:", MODEL_PATH)
print("Audio:", AUDIO_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Audio model not found: {MODEL_PATH}"
    )

if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_PATH}"
    )


# ==========================================
# Load model
# ==========================================

model = load_model(
    MODEL_PATH,
    compile=False
)

print("\n✓ Audio model loaded")


# ==========================================
# Load audio
# ==========================================

audio, sr = librosa.load(
    AUDIO_PATH,
    sr=16000,
    duration=5
)

print("Sample rate:", sr)
print("Audio samples:", len(audio))


# ==========================================
# MFCC extraction
# ==========================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

# Transpose:
# (40, time) → (time, 40)
mfcc = mfcc.T


# ==========================================
# Make exactly 100 time steps
# ==========================================

if mfcc.shape[0] < 100:

    mfcc = np.pad(
        mfcc,
        ((0, 100 - mfcc.shape[0]), (0, 0)),
        mode="constant"
    )

else:

    mfcc = mfcc[:100, :]


print("MFCC shape:", mfcc.shape)


# ==========================================
# Add batch dimension
# ==========================================

features = np.expand_dims(
    mfcc,
    axis=0
)

print("Model input shape:", features.shape)


# ==========================================
# Prediction
# ==========================================

prediction = model.predict(
    features,
    verbose=0
)[0]

predicted_index = np.argmax(prediction)

predicted_emotion = labels[predicted_index]

confidence = prediction[predicted_index] * 100


# ==========================================
# Results
# ==========================================

print("\nPrediction probabilities:")

for label, probability in zip(
    labels,
    prediction
):
    print(
        f"{label:10s}: {probability:.4f}"
    )

print("\n----------------------------")
print("Predicted emotion:", predicted_emotion)
print(f"Confidence: {confidence:.2f}%")
print("----------------------------")