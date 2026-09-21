"""
端侧 MFCC 的 numpy 参考实现
============================================================

本文件是 firmware/HomeGuard/mfcc.c 的等价实现，训练侧与设备侧共用同一套
特征定义。两边只要有一处不一致（帧长、Mel 点数、log 底数、DCT 归一化……），
设备端得到的特征分布就会与训练时不同，精度会直接崩掉。

因此每一步都刻意与 C 代码逐行对应，并由 tools/sim 下的交叉比对脚本对同一段
PCM 做数值验证（`make test`）。这里的向量化只是数学等价改写，不改变结果。

对应关系：
    firmware/HomeGuard/mfcc.c : mfcc_process_frame()
    training/mfcc_ref.py      : compute_mfcc()
"""

import numpy as np

# ---------------- 特征参数（与 mfcc.h 中的宏一一对应） ----------------
SR = 16000                                   # HG_SR
FFT_SIZE = 1024                              # HG_FFT_SIZE
FRAME_LEN = 1024                             # HG_FRAME_LEN
HOP = 512                                    # HG_HOP
N_FRAMES = 32                                # HG_N_FRAMES
N_MEL = 40                                   # HG_N_MEL
N_MFCC = 40                                  # HG_N_MFCC
N_BINS = FFT_SIZE // 2 + 1                   # HG_N_BINS

PRE_EMPH = 0.97                              # mfcc.c: PRE_EMPH
LOG_EPS = 1e-6                               # mfcc.c: LOG_EPS

#: 一次推理窗口覆盖的样点数
WINDOW_SAMPLES = (N_FRAMES - 1) * HOP + FRAME_LEN


def mel_filter_bins() -> np.ndarray:
    """Mel 滤波器组的频点边界（bin 索引），对应 C 侧 mel_filterbank_init()。"""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min, mel_max = hz_to_mel(0.0), hz_to_mel(SR / 2.0)
    idx = np.arange(N_MEL + 2)
    hz = mel_to_hz(mel_min + (mel_max - mel_min) * idx / (N_MEL + 1))
    # C 侧写的是 (int)(hz * FFT_SIZE / SR + 0.5f)，对非负值等价于 floor(x+0.5)
    bins = np.floor(hz * FFT_SIZE / SR + 0.5).astype(int)
    return np.clip(bins, 0, N_BINS - 1)


def _build_filterbank(bins: np.ndarray) -> np.ndarray:
    """把三角滤波器展开成 (N_MEL, N_BINS) 的权重矩阵。

    与 C 侧逐 bin 现算权重完全等价，只是把结果预先算好交给矩阵乘法，
    避免 Python 层的双重循环（否则处理近两千个窗口要跑几十分钟）。
    """
    W = np.zeros((N_MEL, N_BINS), dtype=np.float64)
    for m in range(N_MEL):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, right + 1):
            if center == left:
                w = 1.0 if k <= center else (right - k) / (right - center)
            elif k <= center:
                w = (k - left) / (center - left)
            else:
                w = (right - k) / (right - center)
            W[m, k] = max(w, 0.0)
    return W


_HAMMING = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))
_MEL_BINS = mel_filter_bins()
_FILTERBANK = _build_filterbank(_MEL_BINS)

#: DCT-II 矩阵，对应 C 侧的 cos(pi * i * (m + 0.5) / N_MEL)
_DCT = np.cos(np.pi * np.arange(N_MFCC)[:, None] * (np.arange(N_MEL)[None, :] + 0.5) / N_MEL)


def compute_mfcc(pcm: np.ndarray) -> np.ndarray:
    """对一段 PCM 计算 40 维 x 32 帧 MFCC。

    pcm : int16 一维数组，长度需 >= WINDOW_SAMPLES
    返回 : (N_FRAMES, N_MFCC) 的 float 数组
    """
    pcm = np.asarray(pcm)
    if pcm.size < WINDOW_SAMPLES:
        raise ValueError(f"PCM 长度不足：需要 {WINDOW_SAMPLES}，实际 {pcm.size}")

    x = pcm.astype(np.float64)

    # ---- 分帧：第 k 帧为 x[k*HOP : k*HOP+FRAME_LEN] ----
    idx = np.arange(N_FRAMES)[:, None] * HOP + np.arange(FRAME_LEN)[None, :]
    win = x[idx]                                              # (N_FRAMES, FRAME_LEN)

    # ---- 预加重用的「前一个样点」 ----
    prev = np.empty_like(win)
    prev[:, 0] = x[np.maximum(np.arange(N_FRAMES) * HOP - 1, 0)]
    prev[:, 1:] = win[:, :-1]

    # y[n] = x[n] - 0.97 * x[n-1]
    y = (win - PRE_EMPH * prev) / 32768.0

    # 第 0 帧的第 0 个样点没有前驱，不做差分（与 C 侧 first 分支一致）
    y[0, 0] = win[0, 0] / 32768.0

    # ---- 加 Hamming 窗 ----
    y *= _HAMMING

    # ---- FFT -> 功率谱（单边） ----
    spec = np.abs(np.fft.rfft(y, FFT_SIZE, axis=1)) ** 2       # (N_FRAMES, N_BINS)

    # ---- Mel 滤波 -> log ----
    logmel = np.log(spec @ _FILTERBANK.T + LOG_EPS)            # (N_FRAMES, N_MEL)

    # ---- DCT-II -> MFCC ----
    return logmel @ _DCT.T                                     # (N_FRAMES, N_MFCC)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        raw = np.fromfile(sys.argv[1], dtype="<i2")
        feats = compute_mfcc(raw)
        if len(sys.argv) > 2:
            np.savetxt(sys.argv[2], feats.reshape(-1), fmt="%.6f")
            print(f"已写出 {sys.argv[2]}")
        else:
            print(f"shape={feats.shape} 首帧前5个系数={np.round(feats[0, :5], 4)}")
    else:
        print(f"窗口样点数 WINDOW_SAMPLES = {WINDOW_SAMPLES}")
        print(f"Mel 滤波器边界 = {_MEL_BINS}")
