# Catalyst — Crisis Alert & Threat Analysis with Live Security Trigger

Catalyst is a two-stage audio distress detection pipeline that identifies fear and shouting in real-time or from audio files.

---

## How It Works

### Stage 1 — Distress Detection (YAMNet)
Uses Google's [YAMNet](https://tfhub.dev/google/yamnet/1) model to detect distress-related sounds (screaming, yelling, shouting, crying) in an audio clip.

### Stage 2 — Emotion Classification (CNN)
A 2D-CNN trained on CREMA-D and RAVDESS classifies whether the detected distress is fear-related or not, using MFCC features.

A wake-word detector ("help") runs in parallel via Google Speech Recognition.

---

## Project Structure

```
catalyst/
├── config.py                    # Central config — edit paths here
├── main.py                      # Main entry point (YAMNet pipeline)
├── real-time.py                 # Alternative: sklearn Stage 1 pipeline
│
├── stage1_distress_detector.py  # YAMNet-based distress detector
├── stage2_part_1.py             # Dataset prep (CREMA-D + RAVDESS → .npy)
├── stage2_part_2.py             # CNN training (2D-MFCC path)
├── stage2_emotion_classifier.py # CNN training (1D mean-MFCC path)
│
├── fine_tune.py                 # Fine-tune Stage 2 model on IESC dataset
├── iesc_prep.py                 # Prepare IESC dataset → .npy
├── flatner.py                   # Flatten nested audio folders
├── augment.py                   # Offline fear-sample augmentation
├── convert_audio.py             # Batch resample/normalise audio to 16 kHz WAV
├── data_preparation.py          # Feature extraction utilities
├── plot_accuracy.py             # Training history plots
├── yamnet_classes.py            # Print all YAMNet class IDs
│
├── requirements.txt
└── .gitignore
```

---

## Setup

### 1. Clone & install dependencies
```bash
git clone https://github.com/<your-username>/catalyst.git
cd catalyst
pip install -r requirements.txt
```

### 2. Configure paths
Open `config.py` and set the dataset and model paths to match your local setup:
```python
CREMA_PATH   = "data/crema-d/AudioWAV"
RAVDESS_PATH = "data/ravdess/Flattened_Ravdess"
IESC_RAW_PATH = "data/iesc/raw"
```
Alternatively, set them as environment variables:
```bash
export CREMA_PATH=/path/to/crema-d
export RAVDESS_PATH=/path/to/ravdess
```

---

## Training the Model

### Step 1 — Prepare datasets
```bash
# Flatten IESC (if using nested folder structure)
python flatner.py

# Convert non-WAV audio to 16 kHz WAV
python convert_audio.py

# Augment fear samples offline (optional)
python augment.py

# Extract MFCCs and build training arrays
python stage2_part_1.py
```

### Step 2 — Train Stage 2 classifier
```bash
# 2D-CNN on MFCC frames (recommended)
python stage2_part_2.py

# OR: 1D-CNN on mean-MFCCs
python stage2_emotion_classifier.py
```

### Step 3 — Fine-tune on IESC (optional)
```bash
python iesc_prep.py
python fine_tune.py
```

---

## Running Inference

```bash
python main.py
```

Options at runtime:
- **r** — continuous recording mode (classifies every 2 seconds from microphone)
- **u** — upload/provide a file path for one-shot classification
- **q** — quit

---

## Datasets

| Dataset | Description | Link |
|---|---|---|
| CREMA-D | 7,442 clips from 91 actors, 6 emotions | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |
| RAVDESS | 24 actors, 8 emotions, speech + song | [Zenodo](https://zenodo.org/record/1188976) |
| IESC | Indian Emotional Speech Corpora | Contact dataset authors |

---

## Configuration Reference (`config.py`)

| Key | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `N_MFCC` | `40` | Number of MFCC coefficients |
| `FIXED_LENGTH` | `60` | Fixed time-frame length (pad/truncate) |
| `DISTRESS_CLASS_IDS` | `[6,9,11,19]` | YAMNet class IDs for distress sounds |
| `DISTRESS_THRESHOLD` | `0.1` | YAMNet probability threshold |
| `FEAR_THRESHOLD` | `0.3` | Stage 2 sigmoid threshold for fear |
| `AUGMENT_FACTOR` | `2` | Extra augmented copies per fear sample |
| `EPOCHS_STAGE2` | `17` | Training epochs for Stage 2 CNN |

Run `python yamnet_classes.py` to list all YAMNet class IDs and update `DISTRESS_CLASS_IDS` as needed.

---

## Notes

- `keyboard` library (used for press-to-quit in real-time mode) requires root/admin privileges on Linux. Run with `sudo python main.py` or replace with `input()` prompts.
- Speech recognition requires an internet connection (uses Google Web Speech API).
- All model files and large data folders are excluded from version control via `.gitignore`. Store trained `.keras` / `.pkl` files externally (e.g. Google Drive, HuggingFace Hub) and document the download link here.
