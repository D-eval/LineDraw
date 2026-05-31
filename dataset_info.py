
from linedraw import LineDraw, plot_line

from preprocessor import FeatureExtractor

import librosa
import soundfile as sf
import sys
sys.path.append("../music-detr")
sys.path.append("../dataset")
from spec.cqt import MultiWindowCQT, get_freqs
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from collections import Counter
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pretty_midi

# 3 个数据集
# ../dataset
from read import StackDataset, collate_fn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sr = 44100
min_midi = 24

dataset = StackDataset(sr=sr, min_midi=min_midi, max_midi=107, use_shift=False)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

extractor = FeatureExtractor()


import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

os.makedirs("./experiment", exist_ok=True)

event_counts = []
pitch_cls_counter = Counter()
octave_counter = Counter()

print(f"UnionDataset size: {len(dataset)}")

for audio, target in loader:

    midis = target[0]["midi"]
    if midis=="N":
        midis = []
    # -------------------------
    # Event Count
    # -------------------------

    event_counts.append(
        len(midis)
    )

    # -------------------------
    # Pitch Class
    # -------------------------

    for midi in midis:

        pitch_cls = midi % 12
        octave = midi // 12 - 1

        pitch_cls_counter[pitch_cls] += 1
        octave_counter[octave] += 1


# =====================================
# Event Histogram
# =====================================

plt.figure(figsize=(6,4))

plt.hist(
    event_counts,
    bins=np.arange(
        min(event_counts),
        max(event_counts)+2
    ),
)

plt.xlabel("Number of Notes")
plt.ylabel("Count")
plt.title("Event Count Distribution")

plt.tight_layout()

plt.savefig(
    "./experiment/event_hist.pdf"
)

plt.close()


# =====================================
# Pitch Class Histogram
# =====================================

pitch_names = [
    "C","C#","D","D#",
    "E","F","F#","G",
    "G#","A","A#","B"
]

values = [
    pitch_cls_counter[i]
    for i in range(12)
]

plt.figure(figsize=(8,4))

plt.bar(
    np.arange(12),
    values
)

plt.xticks(
    np.arange(12),
    pitch_names
)

plt.xlabel("Pitch Class")
plt.ylabel("Count")
plt.title("Pitch Class Distribution")

plt.tight_layout()

plt.savefig(
    "./experiment/pitch_cls_hist.pdf"
)

plt.close()


# =====================================
# Octave Histogram
# =====================================

octaves = sorted(
    octave_counter.keys()
)

values = [
    octave_counter[o]
    for o in octaves
]

plt.figure(figsize=(8,4))

plt.bar(
    np.arange(len(octaves)),
    values
)

plt.xticks(
    np.arange(len(octaves)),
    octaves
)

plt.xlabel("Octave")
plt.ylabel("Count")
plt.title("Octave Distribution")

plt.tight_layout()

plt.savefig(
    "./experiment/octave_hist.pdf"
)

plt.close()


# =====================================
# Print Summary
# =====================================

print("========== Dataset Statistics ==========")

print(
    f"Samples: {len(event_counts)}"
)

print(
    f"Avg Notes Per Sample: "
    f"{np.mean(event_counts):.2f}"
)

print(
    f"Min Notes: {np.min(event_counts)}"
)

print(
    f"Max Notes: {np.max(event_counts)}"
)

print()

print(
    "Pitch Class Distribution:"
)

for i in range(12):
    print(
        f"{pitch_names[i]:<2}: "
        f"{pitch_cls_counter[i]}"
    )

print()

print(
    "Octave Distribution:"
)

for octave in octaves:
    print(
        f"{octave}: "
        f"{octave_counter[octave]}"
    )