# LineDraw

Draw sound with lines.

LineDraw is an experimental audio analysis and synthesis framework that represents sound as a collection of sparse frequency trajectories ("lines") in the time-frequency domain.

Instead of reconstructing audio from millions of waveform samples or dense spectrogram bins, LineDraw attempts to explain a spectrogram using a small set of continuous events. Each event is represented as a line:

```text
(t, f, a)
```

where:

* `t` : time
* `f` : frequency bin
* `a` : amplitude

The goal is to minimize both reconstruction error and event complexity.

## Motivation

Traditional audio representations are dense.

For example:

* Waveform: tens of thousands of samples per second
* Spectrogram: thousands of time-frequency bins

However, many musical sounds are highly structured.

A single note can often be represented as:

```text
start time
duration
pitch trajectory
amplitude trajectory
```

instead of hundreds of spectrogram pixels.

LineDraw explores the idea that:

> Sound can be represented as a sparse drawing process.

## Pipeline

```text
Audio
  ↓
Multi-Window CQT
  ↓
Local Frequency Normalization
  ↓
Peak Tracking
  ↓
Line Extraction
  ↓
Event Representation
  ↓
Sine Resynthesis
```

## Local Frequency Normalization

A key component of LineDraw is local frequency normalization:

```text
normalized = spectrogram / local_frequency_energy
```

This removes global frequency response and volume variations, allowing the system to focus on local harmonic structures.

Benefits:

* Amplitude invariance
* Reduced timbre dependence
* Stronger harmonic representation
* Better note tracking

## Event Representation

Each extracted event is represented as a sequence of points:

```python
[
    (t0, f0, a0),
    (t1, f1, a1),
    ...
]
```

with constraints:

```text
f(t+1) ∈ {f(t)-1, f(t), f(t)+1}
```

This forms a continuous frequency trajectory.

## Current Status

Current prototype:

* Multi-window CQT frontend
* Local frequency normalization
* Greedy peak tracking
* Line extraction
* Sine-wave reconstruction
* Logic Pro verification

Future work:

* ADSR envelope fitting
* Harmonic event modeling
* MCTS search
* Reinforcement learning
* Sparse music transcription
* Audio-to-MIDI conversion

## Vision

LineDraw is not intended to be another spectrogram classifier.

The long-term goal is:

> Explain sound using the smallest possible sequence of editable events.

This can be viewed as a sparse reconstruction problem, a minimum description length problem, or a sequential drawing problem.

Ultimately, we hope to build systems that understand sound by learning how to draw it.

# 2026/06/01

大后天考PDE，考完继续写q_learning

感觉这个其实不算 state -> action 的查询

因为首先这个 state 是局部的

也就是当我走到这一步的时候我才查询。

也就是说我们要区分，如果之前有人走到这一步了

以及这一步是新创建的，

这应该被区分为不同的动作。

不过总之还是先撂在这吧，

大后天见。

算了，既然想到了就把 state_action_table 写一下吧