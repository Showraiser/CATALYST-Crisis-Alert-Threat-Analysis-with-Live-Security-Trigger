"""
fine_tune.py — Fine-tune the Stage 2 fear classifier on the IESC dataset.

Loads the pre-trained model from MODEL_PATH, freezes all layers except the
last one, and fine-tunes on the IESC (Indian Emotional Speech Corpora) data
prepared by iesc_prep.py.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import (
    X_IESC_PATH, Y_IESC_PATH, MODEL_PATH, MODEL_FT_PATH,
    TEST_SIZE, RANDOM_STATE, EPOCHS_FINETUNE, BATCH_SIZE, ensure_dirs
)


if __name__ == "__main__":
    ensure_dirs()

    # ── Load IESC Data ─────────────────────────────────────────────────────────
    print("Loading IESC dataset …")
    X_iesc = np.load(X_IESC_PATH, allow_pickle=True)
    y_iesc = np.load(Y_IESC_PATH)
    print(f"  X shape: {X_iesc.shape}  |  y shape: {y_iesc.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_iesc, y_iesc, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Load Pre-trained Model ─────────────────────────────────────────────────
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Pre-fine-tune accuracy: {accuracy:.4f}")

    # ── Freeze All but Last Layer ──────────────────────────────────────────────
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ── Fine-tune ──────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fine-tuned model accuracy: {accuracy:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    ch = input("\nSave the fine-tuned model? (y/n): ").strip().lower()
    if ch == 'y':
        model.save(MODEL_FT_PATH)
        print(f"✅ Fine-tuned model saved → {MODEL_FT_PATH}")
