"""
config.py — Central configuration for Catalyst Distress Detection System.

Edit the paths below to match your local setup before running any script.
"""

import os

# ─── Audio Settings ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000       # Hz — must match YAMNet expectation
N_MFCC = 40               # Number of MFCC coefficients
FIXED_LENGTH = 60         # Fixed number of time frames (pad/truncate target)

# ─── Dataset Paths ─────────────────────────────────────────────────────────────
# CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
CREMA_PATH = os.environ.get("CREMA_PATH", "data/crema-d/AudioWAV")

# RAVDESS (flattened): https://zenodo.org/record/1188976
RAVDESS_PATH = os.environ.get("RAVDESS_PATH", "data/ravdess/Flattened_Ravdess")

# IESC (Indian Emotional Speech Corpora)
IESC_RAW_PATH = os.environ.get("IESC_RAW_PATH", "data/iesc/raw")
IESC_FLAT_PATH = os.environ.get("IESC_FLAT_PATH", "data/iesc/flattened")

# General raw/converted audio folders (for convert_audio.py)
RAW_AUDIO_PATH = os.environ.get("RAW_AUDIO_PATH", "data/raw_audio")
CONVERTED_AUDIO_PATH = os.environ.get("CONVERTED_AUDIO_PATH", "data/converted_audio")

# ─── Processed Dataset Paths (numpy arrays) ────────────────────────────────────
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
X_PATH = os.path.join(PROCESSED_DIR, "X.npy")
Y_PATH = os.path.join(PROCESSED_DIR, "y.npy")
X_IESC_PATH = os.path.join(PROCESSED_DIR, "X_iesc.npy")
Y_IESC_PATH = os.path.join(PROCESSED_DIR, "y_iesc.npy")

# ─── Model Paths ───────────────────────────────────────────────────────────────
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "stage2_fear_classification_model.keras")
MODEL_FT_PATH = os.path.join(MODELS_DIR, "stage2_fear_classification_model_ft.keras")
STAGE1_MODEL_PATH = os.path.join(MODELS_DIR, "stage1_distress_model.pkl")  # for real-time.py

# ─── YAMNet Settings ───────────────────────────────────────────────────────────
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
# YAMNet class indices associated with distress sounds
DISTRESS_CLASS_IDS = [6, 9, 11, 19]  # scream, yell, shout, crying
DISTRESS_THRESHOLD = 0.1

# ─── Training Settings ─────────────────────────────────────────────────────────
AUGMENT_FACTOR = 2    # Extra augmented copies per fear sample
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS_STAGE2 = 17
EPOCHS_FINETUNE = 10
BATCH_SIZE = 32
FEAR_THRESHOLD = 0.3  # Sigmoid probability threshold for fear classification

# ─── Helpers ───────────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [PROCESSED_DIR, MODELS_DIR, CONVERTED_AUDIO_PATH]:
        os.makedirs(d, exist_ok=True)
