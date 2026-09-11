import tensorflow as tf
import numpy as np
import cv2
import pyaudio
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk


# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# ============================
# Load trained models
# ============================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

face_model_path = os.path.join(
    BASE_DIR,
    "FacialModel",
    "emotiondetector1_best.keras"
)

audio_model_path = os.path.join(
    BASE_DIR,
    "AudioModel",
    "best_model_BiLSTM_2D_CNN.keras"
)

print("Face model:", face_model_path)
print("Audio model:", audio_model_path)

# Verify files exist before loading
if not os.path.exists(face_model_path):
    raise FileNotFoundError(f"Face model not found: {face_model_path}")

if not os.path.exists(audio_model_path):
    raise FileNotFoundError(f"Audio model not found: {audio_model_path}")

face_model = load_model(face_model_path, compile=False)
audio_model = load_model(audio_model_path, compile=False)

print("✓ Face model loaded")
print("✓ Audio model loaded")

print("Face input:", face_model.input_shape)
print("Face output:", face_model.output_shape)

print("Audio input:", audio_model.input_shape)
print("Audio output:", audio_model.output_shape)
# Define fusion model
def create_fusion_model(num_emotions=7):
    face_input = Input(shape=(num_emotions,))
    audio_input = Input(shape=(num_emotions,))
    combined = Concatenate()([face_input, audio_input])
    x = Dense(32, activation='relu')(combined)
    x = Dropout(0.3)(x)
    output = Dense(num_emotions, activation='softmax')(x)
    fusion_model = Model(inputs=[face_input, audio_input], outputs=output)
    fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return fusion_model

fusion_model = create_fusion_model(num_emotions=7)
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Face preprocessing (adjusted for typical 48x48 grayscale input)
def preprocess_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))  # Ensure 48x48 as expected by many models
    img = resized.reshape(1, 48, 48, 1) / 255.0  # Normalize to [0, 1]
    return img

# Audio preprocessing (adjusted for consistency)
def preprocess_audio(audio_data, sr=16000, duration=5):
    if len(audio_data) < sr * duration:
        audio_data = np.pad(audio_data, (0, sr * duration - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:sr * duration]
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < 100:
        mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :100]
    mfccs = mfccs.T  # Shape: (100, 40)
    mfccs = np.expand_dims(mfccs, axis=0)  # Shape: (1, 100, 40)
    return mfccs

# Audio recording thread
class AudioRecorder:
    def __init__(self, rate=16000, chunk=1024, duration=5):
        self.rate = rate
        self.chunk = chunk
        self.duration = duration
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(format=pyaudio.paFloat32, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
            print("Microphone initialized successfully.")
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            self.stream = None
        self.audio_queue = queue.Queue()
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.stream is None:
            print("Cannot start recording: Microphone not available.")
            return
        self.running = True
        print("Microphone recording started...")
        threading.Thread(target=self.record, daemon=True).start()

    def record(self):
        frames = []
        start_time = time.time()
        while self.running and (time.time() - start_time) < self.duration:
            try:
                with self.lock:
                    if not self.running:
                        break
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            except Exception as e:
                print(f"Audio recording error: {e}")
                break
        if frames:
            audio_data = np.concatenate(frames)
            self.audio_queue.put(audio_data)
            print("Audio data captured for 5 seconds and queued.")

    def stop(self):
        with self.lock:
            self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Microphone recording stopped.")

# GUI for displaying results (only emotion)
def show_result_gui(pred_label, confidence, modality):
    root = tk.Tk()
    root.title(f"Emotion Recognition - {modality}")
    root.geometry("300x150")
    
    result_label = ttk.Label(root, text=f"Emotion: {pred_label}", font=("Arial", 14))
    result_label.pack(pady=20)
    
    def close_window():
        root.destroy()
    
    root.after(5000, close_window)
    root.mainloop()

# Real-time prediction function with 5-second input
def real_time_emotion_recognition():
    print("Choose input modality:")
    print("1. Face only")
    print("2. Voice only")
    print("3. Both face and voice")
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice not in ['1', '2', '3']:
        print("Invalid choice. Exiting.")
        return

    # Initialize based on choice
    use_face = choice in ['1', '3']
    use_voice = choice in ['2', '3']
    modality = "Face" if choice == '1' else "Voice" if choice == '2' else "Face + Voice"

    # Setup video capture if face is used
    cap = None
    if use_face:
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Webcam opened successfully on index {i}")
                break
            cap.release()
        if not cap or not cap.isOpened():
            print("Error: Could not open webcam.")
            return

    # Setup audio recorder if voice is used
    recorder = None
    if use_voice:
        recorder = AudioRecorder()
        recorder.start()
        if recorder.stream is None:
            print("Exiting due to microphone failure.")
            if use_face:
                cap.release()
            return

    # Load face cascade if needed
    face_cascade = None
    if use_face:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Could not load Haar Cascade classifier.")
            if use_face:
                cap.release()
            if use_voice:
                recorder.stop()
            return

    # Capture input for 5 seconds
    start_time = time.time()
    face_frames = []
    face_pred = None
    audio_pred = None

    print("Capturing input for 5 seconds... Please provide input now.")
    while (time.time() - start_time) < 5:
        if use_face:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)  # Adjusted for better detection
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_frames.append(preprocess_face(face_roi))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Capturing... {:.1f}s".format(5 - (time.time() - start_time)), 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Emotion Recognition", frame)
            cv2.waitKey(1)

    # Stop capturing
    if use_face:
        cap.release()
        cv2.destroyAllWindows()
    if use_voice:
        recorder.stop()

    # Process captured input
    if use_face and face_frames:
        face_preds = [face_model.predict(frame, verbose=0) for frame in face_frames]
        face_pred = np.mean(face_preds, axis=0)  # Average across frames
        print(f"Face prediction probabilities: {face_pred}")
        print(f"Face predicted emotion: {label[face_pred.argmax()]}")
    else:
        print("No face frames captured.")

    if use_voice and not recorder.audio_queue.empty():
        audio_data = recorder.audio_queue.get()
        audio_input = preprocess_audio(audio_data)
        audio_pred = audio_model.predict(audio_input, verbose=0)
        print(f"Audio prediction probabilities: {audio_pred}")
        print(f"Audio predicted emotion: {label[audio_pred.argmax()]}")
    else:
        print("No audio data captured.")

    # Predict emotion based on captured input
    pred_label = "No input detected"
    confidence = 0.0
    if use_face and use_voice and face_pred is not None and audio_pred is not None:
        # Fusion: Use the stronger prediction if one is significantly higher
        face_conf = face_pred.max()
        audio_conf = audio_pred.max()
        if face_conf > audio_conf + 0.2:  # Face dominates if 20% more confident
            pred_label = label[face_pred.argmax()]
            confidence = face_conf
            print(f"Using face prediction: {pred_label}")
        elif audio_conf > face_conf + 0.2:  # Audio dominates
            pred_label = label[audio_pred.argmax()]
            confidence = audio_conf
            print(f"Using audio prediction: {pred_label}")
        else:  # Fusion if close
            final_pred = fusion_model.predict([face_pred, audio_pred], verbose=0)
            pred_label = label[final_pred.argmax()]
            confidence = final_pred.max()
            print(f"Fusion model applied. Final prediction probabilities: {final_pred}")
            print(f"Fusion predicted emotion: {pred_label}")
    elif use_face and face_pred is not None:
        pred_label = label[face_pred.argmax()]
        confidence = face_pred.max()
        print(f"Face-only prediction: {pred_label}")
    elif use_voice and audio_pred is not None:
        pred_label = label[audio_pred.argmax()]
        confidence = audio_pred.max()
        print(f"Voice-only prediction: {pred_label}")

    # Display result in GUI
    print(f"Final emotion to display: {pred_label}")
    show_result_gui(pred_label, confidence, modality)

if __name__ == "__main__":
    real_time_emotion_recognition()