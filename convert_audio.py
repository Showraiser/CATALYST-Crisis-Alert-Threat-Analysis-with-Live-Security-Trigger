"""
convert_audio.py — Batch audio pre-processing utility.

Resamples any audio file to 16 kHz mono, trims / pads to a fixed duration,
normalises amplitude, and saves the result as a 16-bit WAV.
"""

import os
import librosa
import soundfile as sf
from config import RAW_AUDIO_PATH, CONVERTED_AUDIO_PATH, SAMPLE_RATE, ensure_dirs

TARGET_DURATION = 2  # seconds
SUPPORTED_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


def process_audio(file_path, output_path, sr=SAMPLE_RATE, duration=TARGET_DURATION):
    """
    Load, resample, pad/trim, normalise, and save an audio file as WAV.

    Args:
        file_path (str): Input audio path.
        output_path (str): Destination .wav path.
        sr (int): Target sample rate.
        duration (int): Target duration in seconds.
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)

    target_length = sr * duration
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = librosa.util.fix_length(audio, size=target_length)

    max_amp = max(abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    sf.write(output_path, audio, sr)
    print(f"  ✔ {os.path.basename(output_path)}")


def batch_convert(input_folder=RAW_AUDIO_PATH, output_folder=CONVERTED_AUDIO_PATH):
    """Convert all supported audio files in input_folder and save to output_folder."""
    ensure_dirs()
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith(SUPPORTED_EXTS)]

    if not files:
        print(f"No supported audio files found in: {input_folder}")
        return

    print(f"Converting {len(files)} file(s) …")
    for filename in files:
        in_path  = os.path.join(input_folder, filename)
        out_name = os.path.splitext(filename)[0] + '.wav'
        out_path = os.path.join(output_folder, out_name)
        try:
            process_audio(in_path, out_path)
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")

    print(f"\n✅ Done — converted files saved to: {output_folder}")


if __name__ == "__main__":
    batch_convert()
