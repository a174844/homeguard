#!/usr/bin/env python3
"""把 C 实现导出的 MFCC 与 numpy 参考实现逐点比对。

用法:
    python compare_mfcc.py            # 使用同目录下的 test_pcm.bin / mfcc_out.txt
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "training")))

from mfcc_ref import N_FRAMES, N_MFCC, WINDOW_SAMPLES, compute_mfcc  # noqa: E402

PCM = os.path.join(HERE, "test_pcm.bin")
C_OUT = os.path.join(HERE, "mfcc_out.txt")


def main() -> int:
    if not os.path.exists(PCM) or not os.path.exists(C_OUT):
        print("缺少输入：请先运行 gen_test_pcm.py 和 dump_mfcc")
        return 2

    raw = np.fromfile(PCM, dtype="<i2")[:WINDOW_SAMPLES]
    ref = compute_mfcc(raw).reshape(N_FRAMES, N_MFCC)
    got = np.loadtxt(C_OUT).reshape(N_FRAMES, N_MFCC)

    diff = np.abs(ref - got)
    scale = max(float(np.abs(ref).max()), 1e-12)
    rel = float(diff.max()) / scale

    print("=== C 实现 vs numpy 参考实现 ===")
    print(f"规模        : {N_FRAMES} 帧 x {N_MFCC} 维 = {ref.size} 个数值")
    print(f"最大绝对偏差: {diff.max():.3e}")
    print(f"平均绝对偏差: {diff.mean():.3e}")
    print(f"最大相对偏差: {rel:.3e}  (以参考实现最大幅值 {scale:.3f} 为基准)")
    print(f"C   首帧前5维: {np.round(got[0, :5], 6)}")
    print(f"参考 首帧前5维: {np.round(ref[0, :5], 6)}")

    # C 侧用 float32，参考侧用 float64，残差落在 float32 精度量级即视为一致。
    # 能量接近 0 的 Mel 频带经 log() 会放大相对误差，故用相对判据。
    if rel < 1e-3:
        print(f"\n结论: 一致（残差 {rel:.1e} 为 float32 精度量级）")
        return 0
    print("\n结论: 偏差超出 float32 精度可解释范围，需排查 C 实现")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
