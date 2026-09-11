import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# ==========================================
# Project paths
# ==========================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "FacialModel",
    "emotiondetector1_best.keras"
)

IMAGE_PATH = os.path.join(
    PROJECT_DIR,
    "images",
    "test",
    "Sad",
    "32301.png"
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

print("Model path:", MODEL_PATH)
print("Image path:", IMAGE_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ==========================================
# Load model
# ==========================================

model = load_model(
    MODEL_PATH,
    compile=False
)

print("\n✓ Facial model loaded")


# ==========================================
# Read image
# ==========================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise ValueError("Could not read image")


# ==========================================
# Preprocess image
# ==========================================

image = cv2.resize(
    image,
    (48, 48)
)

image = image.astype("float32") / 255.0

# Shape: (48,48) → (1,48,48,1)
image = image.reshape(
    1, 48, 48, 1
)


# ==========================================
# Prediction
# ==========================================

prediction = model.predict(
    image,
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
print(
    f"Confidence: {confidence:.2f}%"
)
print("----------------------------")