我把你最近几个月反复出现、并且已经逐渐收敛到同一个核心目标的想法整理成一个 `future_idea.md` 风格。

---

# future_idea.md

## 最终目标

构建一个能够自动完成音乐理解、扒带、重构与编辑的 Agent。

输入：

```text
wav
```

输出：

```text
音符
和弦
乐器
节拍
混音参数
效果器参数
```

满足：

```text
render(output) ≈ input
```

同时：

```text
动作数量最少
```

即：

```text
频谱误差最小
+
描述长度最小
```

---

# 核心思想

传统 AMT：

```text
Audio
→ Note Label
```

依赖人工标注。

希望改为：

```text
Audio
→ Action Sequence
→ Render
→ Audio
```

利用重构误差训练。

本质上：

```text
Program Synthesis
for Music
```

而不是分类问题。

---

# Phase 1：LineDraw

当前版本。

输入：

```text
CQT
```

提取：

```text
line
```

结构：

```python
(t,f,a,w)
```

其中：

```text
t 时间
f 频率
a 振幅
w 宽度
```

得到：

```text
events
```

再恢复：

```text
wav
```

即：

```text
CQT
→ line
→ wav
```

---

# Phase 2：Amplitude Recovery

当前正在思考。

目标：

```text
events_to_wav(events)
```

之后再经过：

```text
extractor
```

能够恢复原始特征。

即：

```text
extractor(
    render(events)
)
≈
extractor(original)
```

需要研究：

```text
频率响应校正
```

或者：

```text
可微振幅优化
```

形式：

```text
固定结构

优化:
a1,a2,...,an
```

使：

```text
loss=
||feature_rec-feature_gt||
```

最小。

---

# Phase 3：Object Draw

Line 不再是最终对象。

建立层级：

## Level 1

```text
Sinusoid
```

一个正弦波。

---

## Level 2

```text
Note
```

包含：

```text
Fundamental
+
Harmonics
+
Envelope
```

---

## Level 3

```text
Chord
```

例如：

```text
Cmaj7
Dm7
G7
```

---

## Level 4

```text
Instrument Event
```

例如：

```text
Piano Hit
Guitar Stroke
Bass Pluck
```

---

## Level 5

```text
Phrase
```

音乐动机。

---

# Phase 4：Residual Search

当前：

```text
Greedy
```

每次选择最大峰。

未来：

```text
Residual
```

即：

```text
target
-
render(current)
=
residual
```

不断解释剩余部分。

---

# Phase 5：MCTS

状态：

```text
Residual Audio
```

动作：

```text
Add Object
```

例如：

```text
Add Note
Add Chord
Add Drum
```

奖励：

```text
Residual Reduction
-
Action Cost
```

目标：

```text
最少对象
解释音频
```

即：

```text
MDL
(Minimum Description Length)
```

思想。

---

# Phase 6：Differentiable Renderer

建立：

```text
Object
→ Audio
```

可微系统。

例如：

```text
pitch
velocity
attack
release
```

全部可优化。

形式：

```text
Structure Search
+
Gradient Descent
```

即：

```text
MCTS
负责结构

Gradient
负责参数
```

---

# Phase 7：RL Agent

最终形式。

状态：

```text
当前工程
```

动作：

```text
添加音符
删除音符
修改音高
修改力度
修改乐器
修改EQ
修改压缩
```

目标：

```text
render(project)
≈ target
```

奖励：

```text
频谱相似度
-
动作数
```

---

# 长期扩展

## 图像

输入：

```text
Image
```

动作：

```text
画线
画圆
填色
复制
变换
```

目标：

```text
最少绘画操作
恢复图像
```

---

## 视频

动作：

```text
关键帧
运动
形变
```

目标：

```text
最少编辑操作
恢复视频
```

---

## 通用形式

最终统一为：

```text
Observation
↓
Search
↓
Action Sequence
↓
Renderer
↓
Observation
```

优化：

```text
Reconstruction Loss
+
Action Cost
```

本质：

```text
世界模型
+
程序搜索
+
强化学习
```

---

我觉得把你过去几个月所有讨论压缩一下，真正不变的核心其实只有一句话：

> 不通过监督标签，而是通过“最少动作解释观测”，让 Agent 自己学会构建音乐、图像甚至世界的程序表示。

这也是为什么你从 AMT、LineDraw、自动扒带、MCTS、RL、图像重构一路聊下来，看起来换了很多项目，但实际上一直在围绕同一个想法打转。


# 连线策略

$$
Q(s, 线修改操作) <- (1-alpha) * Q(s, 线修改操作) + alpha * (-复杂度增加(事件数) + 频谱更恢复 + gamma * max_下一步线修改操作 * Q(s', a'))
$$