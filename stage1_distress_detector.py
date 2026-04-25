"""
stage1_distress_detector.py — YAMNet-based distress sound detector.

Stage 1 of the Catalyst pipeline: determines whether an audio clip contains
distress-related sounds (screaming, yelling, shouting, crying).
"""

import numpy as np
import librosa
import tensorflow_hub as hub
from config import YAMNET_URL, DISTRESS_CLASS_IDS, DISTRESS_THRESHOLD, SAMPLE_RATE

# Load YAMNet model from TensorFlow Hub (cached after first load)
print("Loading YAMNet model...")
yamnet_model = hub.load(YAMNET_URL)
print("YAMNet loaded.")


def classify_distress(audio_path, threshold=DISTRESS_THRESHOLD):
    """
    Run YAMNet on the given audio file and check if distress is detected.

    Args:
        audio_path (str): Path to the audio file (WAV, MP3, etc.).
        threshold (float): Probability threshold for distress detection.

    Returns:
        tuple: (distress_detected: bool, distress_probs: dict)
            distress_probs maps each distress class ID to its max probability.
    """
    # Load and normalise audio
    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    waveform = waveform.astype("float32")

    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp

    # Run YAMNet — scores shape: (num_frames, num_classes)
    scores, _embeddings, _spectrogram = yamnet_model(waveform)

    # Take max across time frames for each class
    mean_scores = np.max(scores.numpy(), axis=0)

    # Extract distress-related class probabilities
    distress_probs = {cls_id: float(mean_scores[cls_id]) for cls_id in DISTRESS_CLASS_IDS}

    distress_detected = any(prob >= threshold for prob in distress_probs.values())

    return distress_detected, distress_probs


if __name__ == "__main__":
    import sys

    test_audio = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    distress_flag, probs = classify_distress(test_audio, threshold=DISTRESS_THRESHOLD)

    if distress_flag:
        print("🚨 Distress detected!")
    else:
        print("✅ No distress detected.")

    print("Distress class probabilities:")
    for cls_id, prob in probs.items():
        print(f"  Class {cls_id}: {prob:.3f}")
