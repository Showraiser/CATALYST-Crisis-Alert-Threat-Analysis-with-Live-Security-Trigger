"""
real-time.py — Real-time distress detection using the sklearn Stage 1 model.

This is the older pipeline that uses a joblib-serialised sklearn model for
Stage 1 and a Keras model for Stage 2. Kept for reference / comparison.
For the YAMNet-based pipeline use main.py instead.
"""

import os
import tempfile
import librosa
import numpy as np
import joblib
import sounddevice as sd
import soundfile as sf
import keyboard
import speech_recognition as sr
from tensorflow.keras.models import load_model
from config import (
    STAGE1_MODEL_PATH, MODEL_PATH, SAMPLE_RATE, N_MFCC,
    FEAR_THRESHOLD
)

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading Stage 1 distress model …")
distress_model = joblib.load(STAGE1_MODEL_PATH)

print("Loading Stage 2 emotion model …")
emotion_model = load_model(MODEL_PATH)

recognizer = sr.Recognizer()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_audio_file(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio


def extract_mfcc(audio):
    """Return mean-pooled MFCC vector — matches Stage 2 1D-CNN input."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)


def detect_help_word(file_path):
    """Return True if the word 'help' appears in the audio transcript."""
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data).lower()
            print("Transcription:", text)
            return "help" in text
    except Exception as e:
        print(f"Speech recognition failed: {e}")
        return False


# ── Main loop ──────────────────────────────────────────────────────────────────

print("Recording from microphone … Press 'q' to quit.\n")

try:
    while True:
        if keyboard.is_pressed('q'):
            print("Exiting …")
            break

        # Record 2-second chunk
        audio = sd.rec(int(SAMPLE_RATE * 2), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        audio = audio.flatten()

        # Write temp WAV
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, audio, SAMPLE_RATE)

        try:
            audio_data = load_audio_file(temp_path)
            features   = extract_mfcc(audio_data).reshape(1, -1)

            # Stage 1 — distress detection
            is_distress = bool(distress_model.predict(features)[0])
            print("Distress (shouting) detected?", is_distress)

            # Keyword detection
            said_help = detect_help_word(temp_path)
            if said_help:
                print("'Help' word detected in speech!")

            if is_distress or said_help:
                emotion_prob = float(emotion_model.predict(features)[0][0])
                print(f"Emotion score (fear likelihood): {emotion_prob:.3f}")

                if emotion_prob > FEAR_THRESHOLD:
                    print("🚨 Distress Detected (Fear)")
                else:
                    print("⚠ Distress Detected (Other Emotion)")
            else:
                print("✅ No distress detected")

        except Exception as e:
            print(f"Error during classification: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

except KeyboardInterrupt:
    print("\nExiting due to keyboard interrupt.")
