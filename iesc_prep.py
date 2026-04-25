"""
iesc_prep.py — Prepare the IESC (Indian Emotional Speech Corpora) dataset.

Extracts MFCC features, balances fear vs. non-fear classes, adds a CNN
channel dimension, and saves X_iesc.npy / y_iesc.npy for fine_tune.py.
"""

import os
import numpy as np
import librosa
from sklearn.utils import shuffle
from config import (
    IESC_FLAT_PATH, SAMPLE_RATE, N_MFCC, FIXED_LENGTH,
    RANDOM_STATE, X_IESC_PATH, Y_IESC_PATH, ensure_dirs
)


def extract_mfcc(file_path):
    """Load audio and extract a zero-padded / truncated MFCC matrix."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # (time_steps, N_MFCC)

    if mfcc.shape[0] > FIXED_LENGTH:
        mfcc = mfcc[:FIXED_LENGTH, :]
    elif mfcc.shape[0] < FIXED_LENGTH:
        pad_width = FIXED_LENGTH - mfcc.shape[0]
        mfcc = np.vstack((mfcc, np.zeros((pad_width, N_MFCC))))

    return mfcc


def load_iesc_dataset(base_path):
    """
    Walk the flattened IESC directory and return balanced (X, y).

    Labelling: files whose name contains 'fear' (case-insensitive) → 1, else → 0.
    The function balances the two classes by random under-sampling.

    Returns:
        tuple: (X: np.ndarray (N, FIXED_LENGTH, N_MFCC, 1),
                y: np.ndarray (N,))
    """
    X, y = [], []

    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue
            file_path = os.path.join(root, file)
            try:
                mfcc_feat = extract_mfcc(file_path)
                label = 1 if "fear" in file.lower() else 0
                X.append(mfcc_feat)
                y.append(label)
            except Exception as e:
                print(f"  ⚠ Skipped {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Total samples before balancing: {len(y)}")
    print(f"  Fear: {np.sum(y == 1)}  |  Non-fear: {np.sum(y == 0)}")

    # Balance by random under-sampling
    fear_idx     = np.where(y == 1)[0]
    non_fear_idx = np.where(y == 0)[0]
    min_count    = min(len(fear_idx), len(non_fear_idx))

    fear_idx     = np.random.choice(fear_idx,     min_count, replace=False)
    non_fear_idx = np.random.choice(non_fear_idx, min_count, replace=False)

    balanced_idx = np.concatenate((fear_idx, non_fear_idx))
    X = X[balanced_idx]
    y = y[balanced_idx]

    print(f"Balanced dataset size: {len(y)} ({min_count} per class)")

    # Add channel dimension for CNN: (N, 60, 40) → (N, 60, 40, 1)
    X = X[..., np.newaxis]

    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    return X, y


if __name__ == "__main__":
    ensure_dirs()

    X_iesc, y_iesc = load_iesc_dataset(IESC_FLAT_PATH)

    np.save(X_IESC_PATH, X_iesc)
    np.save(Y_IESC_PATH, y_iesc)

    print(f"\n✅ Saved → {X_IESC_PATH}  shape: {X_iesc.shape}")
    print(f"✅ Saved → {Y_IESC_PATH}  shape: {y_iesc.shape}")
