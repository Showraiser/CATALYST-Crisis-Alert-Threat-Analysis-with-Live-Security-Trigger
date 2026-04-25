"""
augment.py — Offline audio augmentation for CREMA-D fear samples.

Reads all FEA (fear) clips from the dataset folder, applies pitch shift,
time stretch, noise, and volume changes, then saves the augmented files
alongside the originals as <original_name>_augmented.wav.
"""

import os
import random
import numpy as np
import librosa
import soundfile as sf
from config import CREMA_PATH, RANDOM_STATE

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ── Augmentation helpers ───────────────────────────────────────────────────────

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return np.clip(audio + noise_factor * noise, -1.0, 1.0)


def change_volume(audio, gain_range=(0.8, 1.2)):
    gain = random.uniform(*gain_range)
    return audio * gain


def augment_sample(audio, sr):
    """Apply pitch shift, time stretch, noise, and volume change."""
    augmented = audio

    n_steps = random.randint(-1, 1)
    augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    rate = random.uniform(0.9, 1.1)
    augmented = librosa.effects.time_stretch(augmented, rate=rate)

    augmented = add_noise(augmented)
    augmented = change_volume(augmented)

    return augmented


# ── Main ───────────────────────────────────────────────────────────────────────

def augment_fear_samples(dataset_folder=CREMA_PATH):
    """
    Walk dataset_folder, augment every FEA clip, and save next to the original.

    Returns a list of (augmented_audio, sr) tuples for downstream use.
    """
    results = []

    for file_name in os.listdir(dataset_folder):
        if not file_name.lower().endswith(".wav"):
            continue

        parts = file_name.split("_")
        if len(parts) < 3 or parts[2] != "FEA":
            continue

        audio_path = os.path.join(dataset_folder, file_name)
        try:
            audio, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"  ⚠ Could not load {file_name}: {e}")
            continue

        augmented = augment_sample(audio, sr)

        base, ext = os.path.splitext(file_name)
        out_name = f"{base}_augmented{ext}"
        out_path = os.path.join(dataset_folder, out_name)
        sf.write(out_path, augmented, sr)

        results.append((augmented, sr))

    print(f"✅ Augmented {len(results)} fear samples → saved to {dataset_folder}")
    return results


if __name__ == "__main__":
    augmented_data = augment_fear_samples()
    print(f"Total augmented samples: {len(augmented_data)}")
