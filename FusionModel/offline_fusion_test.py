import os
import cv2
import numpy as np
import librosa
from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FACE_MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "FacialModel",
    "emotiondetector1_best.keras"
)

AUDIO_MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "AudioModel",
    "best_model_BiLSTM_2D_CNN.keras"
)

FACE_IMAGE_PATH = os.path.join(
    PROJECT_DIR,
    "images",
    "test",
    "Happy",
    "32309.png"
)

AUDIO_PATH = os.path.join(
    PROJECT_DIR,
    "AudioModel",
    "test_audio.wav"
)


# ============================================================
# LABELS
# ============================================================

labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

face_model = load_model(
    FACE_MODEL_PATH,
    compile=False
)

audio_model = load_model(
    AUDIO_MODEL_PATH,
    compile=False
)

print("✓ Face model loaded")
print("✓ Audio model loaded")


# ============================================================
# FACE PREDICTION
# ============================================================

def predict_face(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read face image: {image_path}"
        )

    image = cv2.resize(
        image,
        (48, 48)
    )

    image = image.astype(
        "float32"
    ) / 255.0

    image = image.reshape(
        1, 48, 48, 1
    )

    prediction = face_model.predict(
        image,
        verbose=0
    )[0]

    return prediction


# ============================================================
# AUDIO PREDICTION
# ============================================================

def predict_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        duration=5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # (40, time) -> (time, 40)
    mfcc = mfcc.T

    # Exactly 100 time steps
    if mfcc.shape[0] < 100:

        mfcc = np.pad(
            mfcc,
            (
                (0, 100 - mfcc.shape[0]),
                (0, 0)
            ),
            mode="constant"
        )

    else:

        mfcc = mfcc[:100, :]

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    prediction = audio_model.predict(
        mfcc,
        verbose=0
    )[0]

    return prediction


# ============================================================
# GET PREDICTIONS
# ============================================================

face_pred = predict_face(
    FACE_IMAGE_PATH
)

audio_pred = predict_audio(
    AUDIO_PATH
)


# ============================================================
# DISPLAY INDIVIDUAL PREDICTIONS
# ============================================================

print("\n==============================")
print("FACE PREDICTION")
print("==============================")

for label, probability in zip(
    labels,
    face_pred
):
    print(
        f"{label:10s}: {probability:.4f}"
    )

face_index = np.argmax(face_pred)

print(
    "\nFace emotion:",
    labels[face_index]
)

print(
    f"Face confidence: "
    f"{face_pred[face_index] * 100:.2f}%"
)


print("\n==============================")
print("AUDIO PREDICTION")
print("==============================")

for label, probability in zip(
    labels,
    audio_pred
):
    print(
        f"{label:10s}: {probability:.4f}"
    )

audio_index = np.argmax(audio_pred)

print(
    "\nAudio emotion:",
    labels[audio_index]
)

print(
    f"Audio confidence: "
    f"{audio_pred[audio_index] * 100:.2f}%"
)


# ============================================================
# LATE FUSION
# ============================================================

face_conf = np.max(face_pred)
audio_conf = np.max(audio_pred)

print("\n==============================")
print("FUSION")
print("==============================")

print(
    f"Face confidence : "
    f"{face_conf * 100:.2f}%"
)

print(
    f"Audio confidence: "
    f"{audio_conf * 100:.2f}%"
)

difference = abs(
    face_conf - audio_conf
)

print(
    f"Confidence difference: "
    f"{difference * 100:.2f}%"
)


# Confidence-based dominance
if face_conf > audio_conf + 0.20:

    final_prediction = face_pred
    final_source = "Face"

elif audio_conf > face_conf + 0.20:

    final_prediction = audio_pred
    final_source = "Audio"

else:

    # Weighted probability fusion
    # Used temporarily until the learned fusion
    # network has been trained.
    final_prediction = (
        0.5 * face_pred +
        0.5 * audio_pred
    )

    final_source = "Face + Audio"


# ============================================================
# FINAL RESULT
# ============================================================

final_index = np.argmax(
    final_prediction
)

final_emotion = labels[
    final_index
]

final_confidence = (
    final_prediction[final_index] * 100
)

print("\nFinal prediction probabilities:")

for label, probability in zip(
    labels,
    final_prediction
):
    print(
        f"{label:10s}: {probability:.4f}"
    )

print("\n------------------------------")
print(
    "Final emotion:",
    final_emotion
)
print(
    f"Final confidence: "
    f"{final_confidence:.2f}%"
)
print(
    "Fusion source:",
    final_source
)
print("------------------------------")