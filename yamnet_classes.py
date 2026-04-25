"""
yamnet_classes.py — Print all YAMNet class indices and their display names.

Useful for identifying which class IDs correspond to distress sounds
(screaming, yelling, shouting, crying, etc.) so you can update
DISTRESS_CLASS_IDS in config.py accordingly.
"""

import csv
import tensorflow as tf
import tensorflow_hub as hub
from config import YAMNET_URL

print("Loading YAMNet model …")
model = hub.load(YAMNET_URL)

class_map_path = model.class_map_path().numpy().decode("utf-8")

print(f"\n{'Index':<8} Display Name")
print("-" * 50)

with tf.io.gfile.GFile(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(f"{row['index']:<8} {row['display_name']}")
