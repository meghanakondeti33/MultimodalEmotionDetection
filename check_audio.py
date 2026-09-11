import glob
import librosa

files = glob.glob(
    r"audio_speech_actors_01-24\**\*.wav",
    recursive=True
)

print("Total files:", len(files))

good = 0
bad = 0

for i, path in enumerate(files):
    try:
        audio, sr = librosa.load(path, sr=16000)
        good += 1

        if i < 5:
            print(path)
            print("  Sample rate:", sr)
            print("  Samples:", len(audio))

    except Exception as e:
        bad += 1
        print("FAILED:", path)
        print("ERROR:", e)

print("\n========== RESULT ==========")
print("Good:", good)
print("Bad:", bad)