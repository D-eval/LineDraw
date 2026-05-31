
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

all_threshold = [0.7]
results = []

for j, threshold in enumerate(all_threshold):
    linedraw = LineDraw()
    linedraw.threshold = threshold
    # pitch_cls eval: precision, recall, f1
    # midi eval: precision, recall, f1
    eval_file = "./eval.txt"

    visual_num = 10
    count = 0

    # 结果
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pitch_tp = 0
    total_pitch_fp = 0
    total_pitch_fn = 0

    all_ecr = []
    all_event_num = []
    all_note_num = []

    print(f"UnionDataset size: {len(dataset)}")
    for audio, target in loader:
        print(j, count, len(dataset))
        # print(audio.shape)
        # # (1, 22050, 2)
        # print(target[0]["symbol"], target[0]["midi"])
        # A:min [57, 69, 72, 76] 因为标注本身没有时间信息只有类别信息，所以不考虑时间是否匹配。

        audio = audio[0] # (L, 2)
        L = audio.shape[0]
            
        if target[0]['midi']=="N":
            continue
        
        x, times, freqs, x_ori = extractor(audio)
        freqs = freqs.numpy()
        events, residual = linedraw(x, need_residual=True)
        
        # eval1: eer
        # residual energy ratio, 越小越好
        residual_energy = residual.sum()
        rer = residual_energy / x.sum()
        # Explained Energy Ratio (EER), 越大越好
        ecr = 1 - rer
        
        # eval2: precision, recall, f1
        notelst = linedraw.events2notelst(events, 24) # List[(start, end, midi)]

        pred_midis = {
            midi
            for _,_,midi in notelst
        }

        gt_midis = set(
            target[0]["midi"]
        )

        tp = len(
            pred_midis & gt_midis
        )

        fp = len(
            pred_midis - gt_midis
        )

        fn = len(
            gt_midis - pred_midis
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_ecr.append(
            float(ecr)
        )

        all_event_num.append(
            len(events)
        )

        all_note_num.append(
            len(pred_midis)
        )
        

        pred_pc = {
            midi % 12
            for _,_,midi in notelst
        }

        gt_pc = {
            midi % 12
            for midi in target[0]["midi"]
        }

        total_pitch_tp += len(pred_pc & gt_pc)
        total_pitch_fp += len(pred_pc - gt_pc)
        total_pitch_fn += len(gt_pc - pred_pc)


        # visualize
        if count <= visual_num and threshold==2:
            plot_line(x, events, freqs, times, f"./visualize/line{count}.pdf")
            
            y = 0
            for lines in events:
                y += linedraw.line_to_wave(lines, freqs, L)

            y = y / np.abs(y).max()
            sf.write(
                f'./visualize/syn{count}.mp3',
                y,
                sr,
                format='MP3'
            )
            sf.write(
                f'./visualize/ori{count}.mp3',
                audio,
                sr,
                format='MP3'
            )
            # 这个只能放在 github page 里了
        count += 1

    # export eval.txt

    precision = (
        total_tp /
        (total_tp + total_fp + 1e-9)
    ) # 在 pred 中，有多少正确的

    recall = (
        total_tp /
        (total_tp + total_fn + 1e-9)
    ) # 在 gt 中，有多少正确的

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-9)
    )

    pitch_precision = (
        total_pitch_tp /
        (total_pitch_tp + total_pitch_fp + 1e-9)
    )

    pitch_recall = (
        total_pitch_tp /
        (total_pitch_tp + total_pitch_fn + 1e-9)
    )

    pitch_f1 = (
        2 * pitch_precision * pitch_recall /
        (pitch_precision + pitch_recall + 1e-9)
    )

    mean_ecr = np.mean(
        all_ecr
    )

    mean_event_num = np.mean(
        all_event_num
    )

    mean_note_num = np.mean(
        all_note_num
    )
    
    metric = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pitch_precision": pitch_precision,
        "pitch_recall": pitch_recall,
        "pitch_f1": pitch_f1,
        "ecr": mean_ecr,
        "events": mean_event_num,
        "notes": mean_note_num,
    }
    print(
        f"F1={f1:.4f} "
        f"P={precision:.4f} "
        f"R={recall:.4f} "
        f"PitchF1={pitch_f1:.4f} "
        f"ECR={mean_ecr:.4f}"
    )
    results.append(metric)


import pandas as pd

df = pd.DataFrame(results)

df.to_csv(
    "./experiment/threshold_sweep1.csv",
    index=False
)

# print(df)

# # f1
# plt.figure()

# plt.plot(
#     df.threshold,
#     df.f1,
#     marker="o"
# )

# plt.xlabel("Threshold")
# plt.ylabel("F1")

# plt.savefig(
#     "./experiment/threshold_f1.pdf"
# )

# # ecr

# plt.figure()

# plt.plot(
#     df.threshold,
#     df.ecr,
#     marker="o"
# )

# plt.xlabel("Threshold")
# plt.ylabel("ECR")

# plt.savefig(
#     "./experiment/threshold_ecr.pdf"
# )

# # event count
# plt.figure()

# plt.plot(
#     df.threshold,
#     df.events,
#     marker="o"
# )

# plt.xlabel("Threshold")
# plt.ylabel("Average Events")

# plt.savefig(
#     "./experiment/threshold_events.pdf"
# )

# # pitch_cls f1
# plt.figure()

# plt.plot(
#     df.threshold,
#     df.pitch_f1,
#     marker="o"
# )

# plt.xlabel("Threshold")
# plt.ylabel("Pitch F1")

# plt.savefig(
#     "./experiment/threshold_pitch_f1.pdf"
# )

# # precision-recall
# plt.figure()

# plt.plot(
#     df.recall,
#     df.precision,
#     marker="o"
# )

# plt.xlabel("Recall")
# plt.ylabel("Precision")

# plt.savefig(
#     "./experiment/pr_curve.pdf"
# )

# from sklearn.metrics import auc

# pr_auc = auc(
#     np.sort(df.recall),
#     df.precision.iloc[
#         np.argsort(df.recall)
#     ]
# )

# with open("./experiment/pr-auc.txt", "w") as f:
#     f.write(f"PR-AUC:{pr_auc}")

# print(
#     "PR-AUC:",
#     pr_auc
# )

# # pitch_cls precision-recall
# plt.figure()

# plt.plot(
#     df.pitch_recall,
#     df.pitch_precision,
#     marker="o"
# )

# plt.xlabel("Pitch Recall")
# plt.ylabel("Pitch Precision")

# plt.savefig(
#     "./experiment/pitch_pr_curve.pdf"
# )

# pitch_pr_auc = auc(
#     np.sort(df.pitch_recall),
#     df.pitch_precision.iloc[
#         np.argsort(df.pitch_recall)
#     ]
# )

# with open("./experiment/pitch_pr-auc.txt", "w") as f:
#     f.write(f"Pitch PR-AUC:{pitch_pr_auc}")

# print(
#     "Pitch PR-AUC:",
#     pitch_pr_auc
# )
