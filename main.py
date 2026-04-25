import librosa
import numpy as np
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from tensorflow.keras.models import load_model
from stage1_distress_detector import classify_distress
import keyboard
import time
from config import MODEL_PATH, SAMPLE_RATE, N_MFCC, FIXED_LENGTH

# Load stage 2 emotion model (fear classifier)
emotion_model = load_model(MODEL_PATH)

recognizer = sr.Recognizer()


def extract_mfcc(audio):
    """Extract MFCC features for emotion classifier (must match training)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    # Pad/truncate to fixed length frames
    if mfcc.shape[1] > FIXED_LENGTH:
        mfcc = mfcc[:, :FIXED_LENGTH]
    elif mfcc.shape[1] < FIXED_LENGTH:
        pad_width = FIXED_LENGTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    # Return shape (FIXED_LENGTH, N_MFCC, 1) for CNN
    return mfcc.T[..., np.newaxis]


def detect_help_word(file_path):
    """Detect if the word 'help' was spoken in an audio file."""
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data).lower()
            print("Transcription:", text)
            return "help" in text
    except Exception as e:
        print(f"Speech recognition failed: {e}")
        return False


def process_audio_file_for_distress_emotion(file_path):
    """Run full distress + emotion pipeline on a given audio file."""
    print(f"\nProcessing file: {file_path}")

    # Stage 1: Distress detection via YAMNet
    distress_detected, distress_probs = classify_distress(file_path)
    print("Shouting detected?", distress_detected)
    print("Shouting probabilities:", distress_probs)

    # Load audio for MFCC extraction
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = extract_mfcc(audio)
    mfcc = np.expand_dims(mfcc, axis=0)  # add batch dimension

    # Stage 2: Emotion classification (fear likelihood)
    emotion_prob = emotion_model.predict(mfcc)[0][0]
    print(f"Emotion (fear) probability: {emotion_prob:.3f}")

    if distress_detected and emotion_prob > 0.3:
        print("Distress detected: FEAR (High likelihood)")
    elif distress_detected:
        print("Shout detected: Other emotion")
    elif emotion_prob > 0.3:
        print("Distress detected: FEAR")
    else:
        print("No distress detected")


def continuous_record_mode():
    """Continuously record audio in 2-second chunks and classify distress."""
    print("\nEntering continuous record mode. Press 'q' to quit.")

    duration = 2  # seconds

    while True:
        if keyboard.is_pressed('q'):
            print("Exiting continuous record mode.")
            break

        print("Recording audio for 2 seconds...")
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        audio = audio.flatten()

        # Write to a temporary WAV file
        fd, tmpfile_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmpfile_path, audio, SAMPLE_RATE)

        try:
            distress_detected, distress_probs = classify_distress(tmpfile_path)
            print("Distress detected?", distress_detected)
            print("Distress probabilities:", distress_probs)
            wake_up = detect_help_word(tmpfile_path)

            mfcc = extract_mfcc(audio)
            mfcc = np.expand_dims(mfcc, axis=0)
            emotion_prob = emotion_model.predict(mfcc)[0][0]
            print(f"Emotion (fear) probability: {emotion_prob:.3f}")

            if wake_up:
                print("Distress Detected: Wake up word 'Help'")
            elif distress_detected and emotion_prob > 0.3:
                print("Distress detected: FEAR (High alert)")
            elif emotion_prob > 0.3:
                print("** Distress detected: FEAR **")
            elif distress_detected:
                print("Shout detected: Other emotion")
            else:
                print("No distress detected")
        except Exception as e:
            print(f"Error during classification: {e}")
        finally:
            if os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)

        time.sleep(0.1)


def process_file_mode():
    """Accept a file path from the user and run distress detection on it."""
    file_path = input("Enter path to audio file: ").strip().strip('"')
    if not os.path.isfile(file_path):
        print("File does not exist. Please check the path.")
        return
    process_audio_file_for_distress_emotion(file_path)


def main_loop():
    print("Catalyst — Distress Detection System")
    print("Press 'r' to start continuous recording mode (classify every 2 seconds).")
    print("Press 'u' to input a file path and classify once.")
    print("Press 'q' to quit.")

    while True:
        choice = input("\nYour choice (r/u/q): ").strip().lower()
        if choice == 'r':
            continuous_record_mode()
        elif choice == 'u':
            process_file_mode()
        elif choice == 'q':
            print("Quitting program. Goodbye!")
            break
        else:
            print("Invalid input, please try again.")


if __name__ == "__main__":
    main_loop()
