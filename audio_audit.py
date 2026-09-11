import os
import librosa
import numpy as np
from collections import Counter

ROOT = r"C:\Users\CSE\Desktop\AUDIO DATASETS\Clean_Merged_Dataset"

sample_rates = Counter()
channels = Counter()
durations = []
errors = []
class_counts = Counter()

files = []

for emotion in sorted(os.listdir(ROOT)):
    emotion_dir = os.path.join(ROOT, emotion)

    if not os.path.isdir(emotion_dir):
        continue

    for filename in os.listdir(emotion_dir):
        if filename.lower().endswith(".wav"):
            files.append((emotion, os.path.join(emotion_dir, filename)))

print(f"Total WAV files found: {len(files)}")
print("Starting audio audit...\n")

for i, (emotion, path) in enumerate(files, 1):

    try:
        # Load audio without resampling
        audio, sr = librosa.load(path, sr=None, mono=False)

        # Determine channels
        if audio.ndim == 1:
            channel_count = 1
            duration = len(audio) / sr
        else:
            channel_count = audio.shape[0]
            duration = audio.shape[-1] / sr

        sample_rates[sr] += 1
        channels[channel_count] += 1
        durations.append(duration)
        class_counts[emotion] += 1

    except Exception as e:
        errors.append((path, str(e)))

    if i % 500 == 0:
        print(f"Processed {i}/{len(files)}")

print("\n==============================")
print(" AUDIO DATASET AUDIT")
print("==============================")

print("\nSample Rates:")
for sr, count in sorted(sample_rates.items()):
    print(f"{sr} Hz : {count}")

print("\nChannels:")
for ch, count in sorted(channels.items()):
    print(f"{ch} channel(s) : {count}")

if durations:
    print("\nDuration:")
    print(f"Minimum : {min(durations):.2f} sec")
    print(f"Maximum : {max(durations):.2f} sec")
    print(f"Mean    : {np.mean(durations):.2f} sec")
    print(f"Median  : {np.median(durations):.2f} sec")

print("\nClass counts:")
for emotion, count in sorted(class_counts.items()):
    print(f"{emotion:10s}: {count}")

print("\nUnreadable files:")
print(len(errors))

if errors:
    print("\nFirst 10 errors:")
    for path, error in errors[:10]:
        print(path)
        print(error)

print("\n==============================")