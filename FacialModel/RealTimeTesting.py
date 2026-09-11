import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import load_model

# --------------------------------------------------
# Get the FacialModel directory
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Load model architecture
# --------------------------------------------------
json_path = os.path.join(BASE_DIR, "emotiondetector1.json")

with open(json_path, "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# --------------------------------------------------
# Load trained model weights
# --------------------------------------------------
model_path = os.path.join(BASE_DIR, "emotiondetector1_best.keras")

# Load the complete trained Keras model
model = load_model(model_path)

# --------------------------------------------------
# Load Haar Cascade
# --------------------------------------------------
haar_file = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(haar_file)

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# --------------------------------------------------
# Emotion labels
# --------------------------------------------------
labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# --------------------------------------------------
# Start webcam
# --------------------------------------------------
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press 'q' to quit.")

while True:

    ret, im = webcam.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror image
    im = cv2.flip(im, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Process each detected face
    for (p, q, r, s) in faces:

        image = gray[q:q+s, p:p+r]

        # Draw face rectangle
        cv2.rectangle(
            im,
            (p, q),
            (p+r, q+s),
            (255, 0, 0),
            2
        )

        # Resize face
        image = cv2.resize(image, (48, 48))

        # Prepare input
        img = extract_features(image)

        # Predict emotion
        pred = model.predict(img, verbose=0)

        prediction_label = labels[pred.argmax()]

        # Display emotion
        cv2.putText(
            im,
            prediction_label,
            (p - 10, q - 10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            2,
            (0, 0, 255),
            2
        )

    # Display camera
    cv2.imshow("Multimodal Emotion Detection - Facial", im)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
webcam.release()
cv2.destroyAllWindows()