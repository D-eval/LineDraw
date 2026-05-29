import librosa
import soundfile as sf
import sys
sys.path.append("../music-detr")
sys.path.append("../dataset")
from spec.cqt import MultiWindowCQT, get_freqs
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

min_midi = 24 # 32 Hz
max_midi = 132 # 16k Hz

class FeatureExtractor:
    freqs = get_freqs(min_midi, max_midi)
    preprocessor = MultiWindowCQT(freqs, 44100, 10, 5, stride=0.02) # 2 * 10

    # spec_db = 20 * torch.log10(
    #     freqs + 1e-6
    # )

    def __call__(cls, audio):
        '''audio: (T, 2)'''
        # audio, sr = librosa.load("./output.mp3", mono=False, sr=44100)
        # audio = torch.tensor(audio.T)
        audio = audio.unsqueeze(0)

        # audio *= 10 # 验证幅度不变性

        cqt, _, _ = cls.preprocessor(audio)

        cqt = cqt[0,...] # (T, F, W)

        # T,F,W2 = cqt.shape
        # cqt_all = cqt.reshape(T,F,W2//2,2) # (T, F, W, 2)

        # for pan in range(2):
        #     cqt = cqt_all[..., pan]

        max_idx = cqt.max(-1)[1] # (T, F)

        # print(cqt.shape)
        # print(max_idx.shape)
        # print(max_idx)

        cqt_FW = cqt.mean(0)
        cqt_TF_meanW = cqt.mean(-1) # (T, F)
        cqt_TF_maxW = cqt.max(-1)[0] # (T, F)

        cqt_TF_maxW = cqt_TF_meanW
        # for i in range(10):
        #     cqt_TF_maxW = cqt[...,i] # (T, F)

        # plt.imshow(cqt_TF_meanW.T)
        # plt.show()

        # plt.imshow(cqt_TF_maxW.T)
        # plt.show()

        # 我们真正的，不过是在8度之内的相对能量。
        # 所以，真正的归一化曲线，它的形状，那就是...

        energy_freq = F.avg_pool1d(cqt_TF_maxW.unsqueeze(1), kernel_size=13, padding=13//2, stride=1)
        energy_freq = energy_freq.squeeze(1)

        cqt_TF_norm = cqt_TF_maxW / (energy_freq + 1e-9)
        return cqt_TF_norm
    
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    audio, sr = librosa.load("./output.mp3", mono=False, sr=44100)
    audio = torch.tensor(audio.T)
    cqt_TF_norm = extractor(audio)
    plt.imshow(cqt_TF_norm.T, origin="lower", aspect="auto")
    plt.colorbar()

    plt.savefig(f"./output.pdf")
    # plt.close()
    plt.show()

# 我成功了!

# 或许可以用 mcts 进行搜索
# 只不过直接选取最优的，而不是 visit
