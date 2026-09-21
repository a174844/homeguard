#!/usr/bin/env python3
"""生成确定性的测试 PCM，供 C 实现与 numpy 实现做交叉比对。

刻意不用单一纯音：纯音激励下大部分 Mel 频带能量接近 0，
float32 的 FFT 数值噪声会被 log() 放大并主导结果，成为病态比对条件。
这里用多频点叠加 + 确定性伪随机噪声 + 一个瞬态脉冲，让各个频带都有能量。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "training")))
from mfcc_ref import WINDOW_SAMPLES  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pcm.bin")


def build_signal(n: int = WINDOW_SAMPLES) -> np.ndarray:
    rng = np.random.default_rng(20260221)
    t = np.arange(n) / 16000.0

    # 20 个频点覆盖 200Hz ~ 6850Hz
    acc = np.zeros(n, dtype=np.float64)
    for k in range(20):
        acc += np.sin(2.0 * np.pi * (200.0 + 350.0 * k) * t)

    # 确定性噪声，避免静音频段
    acc += rng.normal(0.0, 1.0, n)

    # 中段加一个瞬态（模拟敲击/碎裂这类冲击声）
    p = n // 2
    acc[p:p + 300] += 8.0 * np.exp(-np.arange(300) / 40.0)

    acc = acc / np.max(np.abs(acc)) * 20000.0
    return np.trunc(acc).astype("<i2")


if __name__ == "__main__":
    sig = build_signal()
    sig.tofile(OUT)
    print(f"已生成 {OUT}: {sig.size} 样点, dtype={sig.dtype}, "
          f"范围 [{sig.min()}, {sig.max()}]")
