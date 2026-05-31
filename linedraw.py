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

class Event:
    start_t: int
    end_t: int
    freq: int
    
    amp: float
    

def _plot_line(x):
    T, F = x.shape
    canvas = np.zeros((T, F))
    for lines in events:
        for t,f,a,w in lines:
            canvas[t, f] = a

    # subplot
    # x: (T, F)
    # canvas: (T, F)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True
    )

    # 原始谱图
    axes[0].imshow(
        x.T,
        origin="lower",
        aspect="auto"
    )
    axes[0].set_title("Input Feature")
    axes[0].set_ylabel("Frequency")

    # 画线结果
    axes[1].imshow(
        canvas.T,
        origin="lower",
        aspect="auto"
    )
    axes[1].set_title(f"Events ({len(events)})")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("./line.pdf")
    # plt.show()

def plot_line(
    x,
    events,
    freqs,
    times,
    save_path="./line.pdf"
):
    import numpy as np
    import matplotlib.pyplot as plt

    T, F = x.shape

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    extent = [
        float(times[0]),
        float(times[-1]),
        0,
        F - 1,
    ]

    im = axes[0].imshow(
        x.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma"
    )

    axes[0].set_title("Normalized Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")

    ax = axes[1]
    ax.set_facecolor("black")

    max_amp = max(
        max(float(a) for _, _, a, _ in line)
        for line in events
    )
    max_amp = max(max_amp, 1e-6)

    for line in events:
        ts = np.array([times[t] for t, f, a, w in line])
        fs = np.array([f for t, f, a, w in line])
        amps = np.array([float(a) for t, f, a, w in line])
        ws = np.array([w for t, f, a, w in line])

        lower = np.maximum(0, fs - ws)
        upper = np.minimum(F - 1, fs + ws)

        alpha = np.clip(amps.mean() / max_amp, 0.15, 1.0)

        ax.fill_between(
            ts,
            lower,
            upper,
            color="white",
            alpha=alpha
        )

        ax.plot(
            ts,
            fs,
            color="white",
            linewidth=0.6,
            alpha=min(1.0, alpha + 0.2)
        )

    ax.set_title(f"LineDraw Representation ({len(events)} events)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    tick_idx = np.arange(0, F, 12)

    for ax in axes:
        ax.set_ylim(0, F - 1)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"{float(freqs[i]):.0f}" for i in tick_idx])

    fig.colorbar(
        im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label="Normalized Spectral Energy"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

class LineDraw:

    threshold = 2
    decay_threshold = 0.7 # 0.3
    width_decay_threshold = 0.3
    radius = 2
    line_min_len = 3
    max_width = 2
    
    # 考虑二阶受迫振动
    # 记录过往的均值
    available_t = 2

    def get_width(cls, residual, temp_t, temp_f, temp_a):
        """
        窄带信号的带宽
        """
        T, F = residual.shape
        width = 0
        temp_max_width = min(cls.max_width, temp_t, F-temp_f-1)
        for w in range(1,temp_max_width+1):
            amp1 = min(residual[temp_t, temp_f+w], residual[temp_t, temp_f-w])
            if amp1 > temp_a * cls.width_decay_threshold:
                width += 1
                temp_a = amp1
            else:
                break
        return width


    def __call__(cls, x, need_residual=False):
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
                                
                start = max(
                    0,
                    temp_t - cls.available_t
                )
                other_a = residual[start:temp_t, candidate].max() # 有时候 a 会震荡
                
                if other_a > temp_a * cls.decay_threshold:
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
                                
                end = min(
                    T,
                    temp_t + cls.available_t + 1
                )
                other_a = residual[
                    temp_t+1:end,
                    candidate
                ].max()
                
                if other_a > temp_a * cls.decay_threshold:
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
            # print(N)
            
            for t,f,a,w in lines:
                # sakuzyo
                l = max(0, f-w)
                r = min(F, f+w+1)
                residual[t,l:r] -= a
                residual.clamp_(min=0)
                
        if need_residual:
            return events, residual
                
        return events


    def line2note(cls,line,min_midi):
        """
        return:
        (start, end, pitch)
        """
        
        start_t = line[0][0]
        end_t = line[-1][0]
        pitches = [
            f
            for _,f,_,_ in line
        ]
        amps = [
            a
            for _,_,a,_ in line
        ]
        f_mode = Counter(
            pitches
        ).most_common(1)[0][0]

        midi = min_midi + f_mode
        return (start_t, end_t, midi)
    def events2notelst(cls,events,min_midi,max_freq=1500):
        result = []
        for line in events:
            temp = cls.line2note(line,min_midi)
            freq = 440 * 2**((temp[2]-69)/12)
            if freq<=max_freq:
                result.append(temp)
        return result


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

 
    def events_to_wav(cls, events):
        y = 0
        for line in events:
            y += cls.line_to_wave(line, freqs, L)

        # y = y / np.abs(y).max()
        # sf.write(
        #     './syn.mp3',
        #     y,
        #     sr,
        #     format='MP3'
        # )
        return y

    @classmethod
    def notelst_export_midi(
        cls,
        notelst,
        times,
        savefile="./output.mid",
        program=0
    ):
        """
        notelst:
            [
                (start_t, end_t, midi),
                ...
            ]
        """

        midi = pretty_midi.PrettyMIDI()

        instrument = pretty_midi.Instrument(
            program=program
        )

        for start_t, end_t, pitch in notelst:

            start = float(
                times[start_t]
            )

            end = float(
                times[end_t]
            )

            note = pretty_midi.Note(
                velocity=100,
                pitch=int(pitch),
                start=start,
                end=end
            )

            instrument.notes.append(
                note
            )

        midi.instruments.append(
            instrument
        )

        midi.write(savefile)

    @staticmethod
    def midi_to_name(midi):

        names = [
            "C","C#","D","D#",
            "E","F","F#","G",
            "G#","A","A#","B"
        ]

        octave = midi // 12 - 1
        note = names[midi % 12]

        return f"{note}{octave}"

    @classmethod
    def line_to_text(
        cls,
        line,
        times,
        min_midi=24,
        mode=3
    ):
        """
        line:
            [(t,f,a,w), ...]

        times:
            (T,)

        mode:

        1:
            full

        2:
            t,f,a

        3:
            summary
        """

        # =====================================
        # mode 1
        # =====================================

        if mode == 1:

            rows = []

            for t, f, a, w in line:

                midi = min_midi + f

                note = cls.midi_to_name(
                    midi
                )

                rows.append(
                    f"{times[t]:.3f}s, "
                    f"{note}, "
                    f"{a:.2f}, "
                    f"{w}"
                )

            return "\n".join(rows)

        # =====================================
        # mode 2
        # =====================================

        elif mode == 2:

            rows = []

            for t, f, a, w in line:

                midi = min_midi + f

                note = cls.midi_to_name(
                    midi
                )

                rows.append(
                    f"{times[t]:.3f}s, "
                    f"{note}, "
                    f"{a:.2f}"
                )

            return "\n".join(rows)

        # =====================================
        # mode 3
        # =====================================

        elif mode == 3:

            start_t = line[0][0]
            end_t = line[-1][0]

            pitches = [
                f
                for _,f,_,_ in line
            ]

            amps = [
                a
                for _,_,a,_ in line
            ]

            f_mode = Counter(
                pitches
            ).most_common(1)[0][0]

            midi = min_midi + f_mode

            note = cls.midi_to_name(
                midi
            )

            avg_amp = float(
                np.mean(amps)
            )

            max_amp = float(
                np.max(amps)
            )

            return (
                f"{times[start_t]:.3f}s"
                f"~"
                f"{times[end_t]:.3f}s, "
                f"{note}, "
                f"avg_amp={avg_amp:.2f}, "
                f"max_amp={max_amp:.2f}, "
                f"len={len(line)}"
            )

        else:

            raise ValueError(
                f"unknown mode={mode}"
            )

    @classmethod
    def events_to_text(
        cls,
        events,
        times,
        min_midi=24,
        mode=3
    ):
        """
        events:
            [
                line1,
                line2,
                ...
            ]

        return:
            str
        """

        texts = []

        for i, line in enumerate(events):

            text = cls.line_to_text(
                line=line,
                times=times,
                min_midi=min_midi,
                mode=mode
            )

            texts.append(
                f"[EVENT {i}]\n{text}"
            )

        return "\n\n".join(texts)


if __name__=="__main__":
    extractor = FeatureExtractor()
    linedraw = LineDraw()

    audio, sr = librosa.load("./output.mp3", mono=False, sr=44100)
    audio = torch.tensor(audio.T) # (L, 2)
    audio = audio.mean(1).unsqueeze(1) # (L,1)
    L = audio.shape[0]
    
    x, times, freqs, x_ori = extractor(audio)
    events = linedraw(x)

    freqs = freqs.numpy()
    
    # visualize
    plot_line(x, events, freqs, times, "./line1.pdf")
    
    # 输入给音频理解大模型，作为中间表示，防止音符识别的幻觉，也可以用于强化学习
    # 也可以作为音频的有损表示，用于训练 music symbol gpt，数据无需标注
    # 还可以用来纠正AMT的标注失误，用于和 line 对比
    text = linedraw.events_to_text(events, times)
    print(text)

    # 导出midi
    notelst = linedraw.events2notelst(events, 24)
    linedraw.notelst_export_midi(notelst, times)
    
   

    # 现在来研究怎么恢复振幅
    # freq_c: (F,)
    freq_c = ... # 求解它，相当于求 lineDraw 这个算子的本征值
    events_new = []
    for line in events:
        line_new = []
        for t,f,a,w in line:
            a_new = a * freq_c[f]
            line_new.append((t,f,a_new,w))
    y = linedraw.events_to_wav(events_new)
    y = y.unsqueeze(1) # (L, 1)
    x_rec, _, _, _ = extractor(y)
    x_rec == x
