"""
flatner.py — Flatten a nested audio dataset into a single directory.

Recursively walks source_root, copies every .wav file into target_folder,
and appends the relative subdirectory path to each filename to avoid
collisions (e.g. actor1/angry/clip.wav → clip_actor1_angry.wav).
"""

import os
import shutil
from config import IESC_RAW_PATH, IESC_FLAT_PATH


def flatten_dataset(source_root=IESC_RAW_PATH, target_folder=IESC_FLAT_PATH):
    """
    Copy all .wav files from a nested directory into a flat folder.

    Args:
        source_root (str): Root of the nested dataset.
        target_folder (str): Destination flat folder.

    Returns:
        int: Number of files copied.
    """
    os.makedirs(target_folder, exist_ok=True)
    count = 0

    for root, _dirs, files in os.walk(source_root):
        for file in files:
            if not file.lower().endswith(".wav"):
                continue

            src_path = os.path.join(root, file)
            relative  = os.path.relpath(root, source_root)
            suffix    = relative.replace(os.sep, "_") if relative != "." else ""

            base, ext = os.path.splitext(file)
            new_name  = f"{base}_{suffix}{ext}" if suffix else file

            dst_path = os.path.join(target_folder, new_name)

            # Avoid overwriting if two files produce the same flat name
            if os.path.exists(dst_path):
                new_name = f"{base}_{suffix}_{count}{ext}"
                dst_path = os.path.join(target_folder, new_name)

            shutil.copy2(src_path, dst_path)
            count += 1

    print(f"✅ Copied {count} .wav files → {target_folder}")
    return count


if __name__ == "__main__":
    flatten_dataset()
