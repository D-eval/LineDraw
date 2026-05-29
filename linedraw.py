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

class Event:
    start_t: int
    end_t: int
    freq: int
    
    amp: float
    

# line: [(t0,f0),(t1,f1),...] f{i+1} \in {f{i}-1, f{i}, f{i}+1}


class LineDraw:

    threshold = 2
    decay_threshold = 0.3
    radius = 2
    line_min_len = 1
    max_width = 2

    def get_width(cls, residual, temp_t, temp_f, temp_a):
        """
        窄带信号的带宽
        """
        T, F = residual.shape
        width = 0
        temp_max_width = min(cls.max_width, temp_t, F-temp_f-1)
        for w in range(1,temp_max_width+1):
            amp1 = min(residual[temp_t, temp_f+w], residual[temp_t, temp_f-w])
            if amp1 > temp_a * cls.decay_threshold:
                width += 1
                temp_a = amp1
            else:
                break
        return width


    def __call__(cls, x):
        T,F = x.shape
        residual = x.clone()
        events = []
        # exclusion = []
        N = 0
        while 1:
            t0, f0 = torch.nonzero(
                residual == residual.max()
            )[0]
            t0 = t0.item()
            f0 = f0.item()
            amp0 = residual[t0,f0].item()
            if amp0 < cls.threshold:
                break
            # if (t0, f0) in exclusion:
            #     break
            
            # width
            temp_t = t0
            temp_f = f0
            temp_a = amp0
            temp_w = cls.get_width(residual, temp_t, temp_f, temp_a)
            lines = [(temp_t, temp_f, temp_a, temp_w)]

            # left expand
            temp_t = t0
            temp_f = f0
            temp_a = amp0
            while temp_t > 0:
                candidate = torch.arange(max(temp_f-cls.radius, 0), min(temp_f+cls.radius, F))
                idx = residual[temp_t-1, candidate].argmax().item()
                new_f = candidate[idx].item()
                new_a = residual[temp_t-1, new_f]
                if new_a > temp_a * cls.decay_threshold:
                    temp_t -= 1
                    temp_f = new_f
                    temp_a = new_a
                    temp_w = cls.get_width(residual, temp_t, temp_f, temp_a)
                    lines = [(temp_t, temp_f, temp_a.item(), temp_w)] + lines
                else:
                    break
            
            # right expand
            temp_t,temp_f,temp_a,_ = lines[-1]
            while temp_t < T-1:
                candidate = torch.arange(max(temp_f-cls.radius+1, 0), min(temp_f+cls.radius, F))
                idx = residual[temp_t+1, candidate].argmax().item()
                new_f = candidate[idx].item()
                new_a = residual[temp_t+1, new_f]
                if new_a > temp_a * cls.decay_threshold:
                    temp_t += 1
                    temp_f = new_f
                    temp_a = new_a
                    temp_w = cls.get_width(residual, temp_t, temp_f, temp_a)
                    lines = lines + [(temp_t, temp_f, temp_a.item(), temp_w)]
                else:
                    break
                
            if len(lines)<=cls.line_min_len:
                # exclusion.append((t0, f0))
                residual[t0,f0] = 0
                continue
                
            # get event
            events.append(lines)
            N += 1
            print(N)
            
            for t,f,a,w in lines:
                # sakuzyo
                l = max(0, f-w)
                r = min(F, f+w+1)
                residual[t,l:r] -= a
                residual.clamp_(min=0)
        return events

    def line_to_wave(
        cls,
        lines,
        freqs,
        total_samples,
        stride=0.02,
        sr=44100,
    ):
        """
        lines:
            [(t,f,a,w), ...]

        return:
            mono waveform (N,)
        """
        if len(lines) == 0:
            return np.zeros(1, dtype=np.float32)

        y = np.zeros(
            total_samples,
            dtype=np.float32
        )

        phase = 0.0

        for t, f, a, w in lines:

            start = int(
                t * stride * sr
            )

            end = int(
                (t + 1) * stride * sr
            )

            freq = freqs[f]

            n = end - start

            ts = np.arange(n) / sr

            seg = (
                a *
                np.sin(
                    phase +
                    2*np.pi*freq*ts
                )
            )

            y[start:end] += seg

            phase += (
                2*np.pi*freq*n/sr
            )

        return y


if __name__=="__main__":
    extractor = FeatureExtractor()
    linedraw = LineDraw()

    audio, sr = librosa.load("./output.mp3", mono=False, sr=44100)
    audio = torch.tensor(audio.T) # (L, 2)
    L = audio.shape[0]
    
    x = extractor(audio)
    events = linedraw(x)

    freqs = extractor.freqs.numpy()
    y = 0
    for lines in events:
        y += linedraw.line_to_wave(lines, freqs, L)

    y = y / np.abs(y).max()
    sf.write(
        './syn.mp3',
        y,
        sr,
        format='MP3'
    )
    # # x (T, F)
    # T, F = x.shape


    # canvas = np.zeros((T, F))
    # for lines in events:
    #     for t,f,a,w in lines:
    #         canvas[t, f] = a

    # # subplot
    # # x: (T, F)
    # # canvas: (T, F)
    # fig, axes = plt.subplots(
    #     2,
    #     1,
    #     figsize=(12, 8),
    #     sharex=True
    # )

    # # 原始谱图
    # axes[0].imshow(
    #     x.T,
    #     origin="lower",
    #     aspect="auto"
    # )
    # axes[0].set_title("Input Feature")
    # axes[0].set_ylabel("Frequency")

    # # 画线结果
    # axes[1].imshow(
    #     canvas.T,
    #     origin="lower",
    #     aspect="auto"
    # )
    # axes[1].set_title(f"Events ({len(events)})")
    # axes[1].set_xlabel("Time")
    # axes[1].set_ylabel("Frequency")

    # plt.tight_layout()
    # plt.savefig("./line.pdf")
    # plt.show()