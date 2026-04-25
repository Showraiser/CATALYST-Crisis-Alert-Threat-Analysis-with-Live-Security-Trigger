"""
data_preparation.py — Feature extraction utilities for Stage 1 and Stage 2.

Used directly by stage2_emotion_classifier.py (mean-MFCC path).
"""

import os
import numpy as np
import librosa
from tqdm import tqdm
from config import CREMA_PATH, SAMPLE_RATE, N_MFCC

# ── Emotion label maps ─────────────────────────────────────────────────────────
EMOTION_MAP = {
    "ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5
}

FEAR_MAP = {
    "FEA": 1,
    "ANG": 0, "DIS": 0, "HAP": 0, "NEU": 0, "SAD": 0,
}


def extract_features(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extract mean-pooled MFCC features from a WAV file.

    Returns a 1-D vector of shape (n_mfcc,).
    """
    y, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def parse_filename(filename):
    """
    Extract emotion code and intensity level from a CREMA-D filename.

    CREMA-D format: <actorID>_<sentenceID>_<emotion>_<level>.wav
    Returns (emotion: str, intensity: str).
    """
    parts = os.path.splitext(filename)[0].split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[2], parts[3]  # emotion, intensity


def create_dataset(audio_dir=CREMA_PATH, stage=1):
    """
    Build a feature matrix and label vector from CREMA-D audio files.

    Args:
        audio_dir (str): Path to the folder containing .wav files.
        stage (int): 1 — distress vs. calm binary labels;
                     2 — fear vs. non-fear binary labels.

    Returns:
        tuple: (X: np.ndarray (N, N_MFCC), y: np.ndarray (N,))
    """
    X, y = [], []

    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    for filename in tqdm(wav_files, desc=f"Stage {stage} data prep"):
        try:
            emotion, intensity = parse_filename(filename)
        except ValueError as e:
            print(f"  ⚠ Skipped ({e})")
            continue

        # Stage 1: only high-intensity samples
        if stage == 1 and intensity != "HI":
            continue

        # Stage 2: skip unknown emotion codes
        if stage == 2 and emotion not in FEAR_MAP:
            continue

        path = os.path.join(audio_dir, filename)
        try:
            features = extract_features(path)
        except Exception as e:
            print(f"  ⚠ Skipped {path}: {e}")
            continue

        if stage == 1:
            label = 1 if emotion in ("ANG", "FEA") else 0
        else:
            label = FEAR_MAP[emotion]

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)
