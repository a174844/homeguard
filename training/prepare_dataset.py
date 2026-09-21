#!/usr/bin/env python3
"""把原始 wav 数据集转换成训练用的 MFCC 特征集。

关键点：
1. 数据集是 44100Hz，而设备端麦克风按 16000Hz 采样。
   这里统一重采样到 16000Hz，保证训练与推理的时频尺度一致。
2. 特征提取完全复用 training/mfcc_ref.py —— 也就是与设备端 mfcc.c 等价的那一份实现。
   绝不在训练侧另用 librosa 之类的库提取特征：那样两边参数一旦不同，
   设备上精度会莫名下降且极难排查。
3. 划分 train/val/test 时按「文件」划分，不是按窗口划分。
   同一个 5 秒文件切出的窗口高度相关，若按窗口随机划分会造成数据泄漏，
   测试集准确率会虚高。

用法:
    python prepare_dataset.py --data /path/to/dataset --out dataset.npz
"""

import argparse
import os
import sys
import wave

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mfcc_ref import (  # noqa: E402
    N_FRAMES, N_MFCC, SR, WINDOW_SAMPLES, compute_mfcc,
)

#: 切片步长（样点）。1.056s 的窗口在 5s 文件上按 0.25s 步长滑动，每文件约 16 个窗口。
DEFAULT_STRIDE = 4000


def read_wav_16k(path: str) -> np.ndarray:
    """读取 wav 并重采样到 SR，返回 int16 数组。"""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: 仅支持 16bit PCM")
        ch = w.getnchannels()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)

    if sr != SR:
        # 16000 / 44100 = 160 / 441，resample_poly 内部带抗混叠滤波
        g = np.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)

    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def slice_windows(x: np.ndarray, stride: int) -> np.ndarray:
    """把一段长音频切成若干 WINDOW_SAMPLES 长的窗口。"""
    if x.size < WINDOW_SAMPLES:
        return np.empty((0, WINDOW_SAMPLES), dtype=np.int16)
    starts = range(0, x.size - WINDOW_SAMPLES + 1, stride)
    return np.stack([x[s:s + WINDOW_SAMPLES] for s in starts])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="数据集根目录，一级子目录为类别名")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.npz"))
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--seed", type=int, default=20260221)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
    if not classes:
        print(f"数据集目录下没有类别子目录: {args.data}")
        return 2
    print(f"类别: {classes}")

    feats, labels, file_ids = [], [], []
    file_list = []

    for ci, cname in enumerate(classes):
        cdir = os.path.join(args.data, cname)
        wavs = sorted(f for f in os.listdir(cdir) if f.lower().endswith(".wav"))
        n_win = 0
        for fn in wavs:
            fid = len(file_list)
            file_list.append(f"{cname}/{fn}")
            try:
                x = read_wav_16k(os.path.join(cdir, fn))
            except Exception as e:
                print(f"  跳过 {cname}/{fn}: {e}")
                file_list.pop()
                continue
            for w in slice_windows(x, args.stride):
                feats.append(compute_mfcc(w))
                labels.append(ci)
                file_ids.append(fid)
                n_win += 1
        print(f"  {cname:<12} {len(wavs):>3} 个文件 -> {n_win:>4} 个窗口")

    X = np.asarray(feats, dtype=np.float32)          # (N, 32, 40)
    y = np.asarray(labels, dtype=np.int64)
    fid = np.asarray(file_ids, dtype=np.int64)
    print(f"\n总窗口数: {X.shape[0]}, 特征形状: {X.shape[1:]}")

    # ---- 按文件划分，避免同一文件的不同窗口同时出现在训练集与测试集 ----
    rng = np.random.default_rng(args.seed)
    n_files = len(file_list)
    perm = rng.permutation(n_files)
    n_test = max(1, int(round(n_files * args.test_ratio)))
    n_val = max(1, int(round(n_files * args.val_ratio)))
    test_files = set(perm[:n_test].tolist())
    val_files = set(perm[n_test:n_test + n_val].tolist())

    mask_test = np.isin(fid, list(test_files))
    mask_val = np.isin(fid, list(val_files))
    mask_train = ~(mask_test | mask_val)

    for name, m in [("train", mask_train), ("val", mask_val), ("test", mask_test)]:
        cls_cnt = np.bincount(y[m], minlength=len(classes))
        print(f"  {name:<5}: {m.sum():>5} 个窗口, 各类 {cls_cnt.tolist()}")

    # ---- 标准化统计量只在训练集上计算 ----
    mean = X[mask_train].reshape(-1, N_MFCC).mean(axis=0).astype(np.float32)
    std = X[mask_train].reshape(-1, N_MFCC).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1e-6

    np.savez_compressed(
        args.out,
        x_train=X[mask_train], y_train=y[mask_train],
        x_val=X[mask_val], y_val=y[mask_val],
        x_test=X[mask_test], y_test=y[mask_test],
        mean=mean, std=std,
        classes=np.array(classes),
        test_files=np.array(sorted(file_list[i] for i in test_files)),
    )
    print(f"\n已保存 {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
