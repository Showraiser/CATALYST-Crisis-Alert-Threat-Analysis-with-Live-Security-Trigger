"""
stage2_emotion_classifier.py — 1D-CNN emotion classifier (fear vs. non-fear).

Alternative to stage2_part_2.py: uses mean-pooled MFCCs (1-D feature vectors)
instead of 2-D MFCC frames, and trains a 1D-CNN with class-weight balancing.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from collections import Counter
from data_preparation import create_dataset
from plot_accuracy import plot_accuracy
from config import (
    CREMA_PATH, MODEL_PATH, TEST_SIZE, RANDOM_STATE,
    BATCH_SIZE, FEAR_THRESHOLD, ensure_dirs
)


def build_model(n_features):
    """Build a 1D-CNN for binary fear classification on mean-MFCC vectors."""
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((n_features, 1), input_shape=(n_features,)),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
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

    # ── Load Dataset ───────────────────────────────────────────────────────────
    print("Loading dataset (stage 2) …")
    X, y = create_dataset(audio_dir=CREMA_PATH, stage=2)
    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train distribution: {Counter(y_train)}")
    print(f"  Test  distribution: {Counter(y_test)}")

    # ── Class Weights ──────────────────────────────────────────────────────────
    class_weights_arr = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights_arr))
    print("Class weights:", class_weights_dict)

    # ── Build & Train ──────────────────────────────────────────────────────────
    model = build_model(n_features=X_train.shape[1])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = (model.predict(X_test) > FEAR_THRESHOLD).astype("int32")
    print(classification_report(y_test, y_pred, target_names=["Non-Fear", "Fear"]))

    plot_accuracy(history)

    # ── Save ───────────────────────────────────────────────────────────────────
    ch = input("\nSave the model? (y/n): ").strip().lower()
    if ch == 'y':
        model.save(MODEL_PATH)
        print(f"✅ Model saved → {MODEL_PATH}")
