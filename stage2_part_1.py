"""
stage2_part_1.py — Dataset preparation for Stage 2 (fear vs. non-fear).

Loads CREMA-D and RAVDESS datasets, extracts MFCC features, balances classes,
augments fear samples, and saves X.npy / y.npy for use in stage2_part_2.py.
"""

import os
import numpy as np
import librosa
import random
from sklearn.utils import shuffle
from config import (
    CREMA_PATH, RAVDESS_PATH, SAMPLE_RATE, N_MFCC, FIXED_LENGTH,
    AUGMENT_FACTOR, RANDOM_STATE, X_PATH, Y_PATH, ensure_dirs
)

# ─── Reproducibility ───────────────────────────────────────────────────────────
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

FEAR_LABEL = 1
NON_FEAR_LABEL = 0


# ─── Audio Augmentation ────────────────────────────────────────────────────────

def add_noise(audio):
    noise_amp = 0.005 * np.random.uniform() * np.amax(np.abs(audio) + 1e-9)
    return audio + noise_amp * np.random.normal(size=audio.shape)


def shift_audio(audio):
    shift_range = int(np.random.uniform(-0.1, 0.1) * len(audio))
    return np.roll(audio, shift_range)


def stretch_audio(audio, rate=1.0):
    return librosa.effects.time_stretch(audio, rate=rate)


def augment_audio(audio):
    choice = random.choice(["noise", "shift", "stretch"])
    if choice == "noise":
        return add_noise(audio)
    elif choice == "shift":
        return shift_audio(audio)
    elif choice == "stretch":
        rate = np.random.uniform(0.8, 1.2)
        return stretch_audio(audio, rate)
    return audio


# ─── Feature Extraction ────────────────────────────────────────────────────────

def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Extract MFCC and return shape (time_steps, n_mfcc)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T


def pad_truncate(mfcc, max_len=FIXED_LENGTH):
    """Pad with zeros or truncate MFCC to a fixed number of time frames."""
    if mfcc.shape[0] > max_len:
        return mfcc[:max_len, :]
    pad_width = max_len - mfcc.shape[0]
    return np.vstack((mfcc, np.zeros((pad_width, mfcc.shape[1]))))


# ─── Dataset Loader ────────────────────────────────────────────────────────────

def load_dataset(base_path, dataset_type):
    """
    Walk a dataset directory and return (X, y).

    Args:
        base_path (str): Root folder containing .wav files.
        dataset_type (str): 'crema' or 'ravdess' — controls label parsing.

    Returns:
        tuple: (X: np.ndarray of shape (N, FIXED_LENGTH, N_MFCC),
                y: np.ndarray of shape (N,))
    """
    X, y = [], []
    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue
            file_path = os.path.join(root, file)
            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                if dataset_type == "crema":
                    # CREMA-D filename: <actorID>_<sentence>_<emotion>_<level>.wav
                    parts = file.split("_")
                    if len(parts) < 3:
                        continue
                    emotion = parts[2]
                    label = FEAR_LABEL if emotion == "FEA" else NON_FEAR_LABEL

                elif dataset_type == "ravdess":
                    # RAVDESS filename: 03-01-<emotion_code>-...wav (1-indexed)
                    parts = os.path.splitext(file)[0].split("-")
                    if len(parts) < 3:
                        continue
                    emotion_code = int(parts[2])
                    label = FEAR_LABEL if emotion_code == 6 else NON_FEAR_LABEL

                else:
                    label = NON_FEAR_LABEL

                mfcc_feat = extract_mfcc(audio, sr)
                mfcc_feat = pad_truncate(mfcc_feat, FIXED_LENGTH)
                # Z-score normalise per sample
                mfcc_feat = (mfcc_feat - np.mean(mfcc_feat)) / (np.std(mfcc_feat) + 1e-9)

                X.append(mfcc_feat)
                y.append(label)

            except Exception as e:
                print(f"  ⚠ Skipped {file_path}: {e}")

    return np.array(X), np.array(y)


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_dirs()

    print("Loading CREMA-D …")
    X_crema, y_crema = load_dataset(CREMA_PATH, "crema")
    print(f"  CREMA-D: {len(y_crema)} samples")

    print("Loading RAVDESS …")
    X_ravdess, y_ravdess = load_dataset(RAVDESS_PATH, "ravdess")
    print(f"  RAVDESS: {len(y_ravdess)} samples")

    # Combine
    X = np.concatenate((X_crema, X_ravdess), axis=0)
    y = np.concatenate((y_crema, y_ravdess), axis=0)

    print(f"\nTotal samples: {len(y)}")
    print(f"  Fear: {np.sum(y == FEAR_LABEL)}  |  Non-fear: {np.sum(y == NON_FEAR_LABEL)}")

    # Balance
    fear_idx = np.where(y == FEAR_LABEL)[0]
    non_fear_idx = np.where(y == NON_FEAR_LABEL)[0]
    min_count = min(len(fear_idx), len(non_fear_idx))

    fear_idx_sel = np.random.choice(fear_idx, min_count, replace=False)
    non_fear_idx_sel = np.random.choice(non_fear_idx, min_count, replace=False)

    balanced_idx = np.concatenate((fear_idx_sel, non_fear_idx_sel))
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]

    print(f"Balanced dataset: {len(y_balanced)} samples ({min_count} per class)")

    # Augment fear samples (add small noise to MFCC features)
    print(f"Augmenting fear samples (factor {AUGMENT_FACTOR}) …")
    X_aug, y_aug = [], []
    for idx in fear_idx_sel:
        for _ in range(AUGMENT_FACTOR):
            noisy = X[idx] + np.random.normal(0, 0.01, X[idx].shape)
            X_aug.append(noisy)
            y_aug.append(FEAR_LABEL)

    X_final = np.concatenate((X_balanced, np.array(X_aug)), axis=0)
    y_final = np.concatenate((y_balanced, np.array(y_aug)), axis=0)

    X_final, y_final = shuffle(X_final, y_final, random_state=RANDOM_STATE)

    print(f"Final dataset size: {len(y_final)}")

    np.save(X_PATH, X_final)
    np.save(Y_PATH, y_final)
    print(f"✅ Saved → {X_PATH}")
    print(f"✅ Saved → {Y_PATH}")
