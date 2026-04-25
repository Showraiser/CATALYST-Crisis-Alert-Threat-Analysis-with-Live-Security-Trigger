"""
stage2_part_2.py — CNN model training for Stage 2 (fear vs. non-fear).

Loads X.npy / y.npy produced by stage2_part_1.py, trains a 2D-CNN,
evaluates it, and optionally saves the model.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
from plot_accuracy import plot_accuracy
from config import (
    X_PATH, Y_PATH, MODEL_PATH, FIXED_LENGTH, N_MFCC,
    TEST_SIZE, RANDOM_STATE, EPOCHS_STAGE2, BATCH_SIZE, ensure_dirs
)


def pad_or_truncate(mfcc, max_len=FIXED_LENGTH):
    """Pad with zeros or truncate MFCC to a fixed number of time frames."""
    if mfcc.shape[0] > max_len:
        return mfcc[:max_len, :]
    elif mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        return np.vstack((mfcc, np.zeros((pad_width, mfcc.shape[1]))))
    return mfcc


def build_model(input_shape):
    """Build and compile a 2D-CNN for binary fear classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    ensure_dirs()

    # ── Load Data ──────────────────────────────────────────────────────────────
    print("Loading dataset …")
    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH)
    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")

    # ── Pad / Truncate ─────────────────────────────────────────────────────────
    print("Padding / truncating MFCCs to fixed length …")
    X_fixed = np.array([pad_or_truncate(x) for x in X])
    print(f"  After padding: {X_fixed.shape}")

    # ── Shuffle & Split ────────────────────────────────────────────────────────
    X_fixed, y = shuffle(X_fixed, y, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X_fixed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Add channel dimension: (N, 60, 40) → (N, 60, 40, 1)
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn  = X_test[..., np.newaxis]
    print(f"  Train: {X_train_cnn.shape}  |  Test: {X_test_cnn.shape}")

    # ── Build & Train ──────────────────────────────────────────────────────────
    model = build_model(input_shape=(FIXED_LENGTH, N_MFCC, 1))
    model.summary()

    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=EPOCHS_STAGE2,
        batch_size=BATCH_SIZE,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test_cnn)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Non-Fear", "Fear"]))

    plot_accuracy(history)

    # ── Save ───────────────────────────────────────────────────────────────────
    ch = input("\nSave the model? (y/n): ").strip().lower()
    if ch == 'y':
        model.save(MODEL_PATH)
        print(f"✅ Model saved → {MODEL_PATH}")
