#!/usr/bin/env python3
"""PC 端对照验证工具：用同一套 INT8 模型跑一遍固件上的完整推理流水。

为什么要这个脚本
============================================================
设备上出现「明明有声音却不报」或者「一直误报」时，最难判断的是问题出在哪一层：
I2S 采到的样点不对？MFCC 算错？还是量化/决策的问题？

本脚本把设备端的流水线在 PC 上完整重现一遍：

    WAV -> 重采样到 16k -> 流式 MFCC(32x40) -> 标准化 -> INT8 量化
        -> tflite INT8 模型 -> 反量化 -> 阈值 + 连续确认

其中 MFCC 用的是 training/mfcc_ref.py（固件 mfcc.c 的等价实现），
量化参数与标准化常量直接从 firmware/HomeGuard/model_data.h 里解析，
不从别处复制 —— 保证和设备端编译的是同一组数。

这样就得到一个对照基线：如果 PC 端跑同一段音频能识别出来，
说明模型没问题，要去查麦克风/采样链路；如果两边都不行，那就是模型本身的局限。

用法
============================================================
    python verify_wav.py test.wav
    python verify_wav.py test.wav --json

依赖：numpy、tensorflow（或 tflite-runtime）、可选 scipy（用于重采样）
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))

import mfcc_ref  # noqa: E402

MODEL_H = os.path.join(ROOT, "firmware", "HomeGuard", "model_data.h")
TFLITE = os.path.join(ROOT, "training", "model_int8.tflite")


# ---------------------------------------------------------------
# 从 model_data.h 里解析常量
# ---------------------------------------------------------------

def _nums(name):
    with open(MODEL_H, encoding="utf-8") as f:
        src = f.read()
    m = re.search(rf"{name}\s*\[\s*\d+\s*\]\s*=\s*\{{(.*?)\}}", src, re.S)
    if not m:
        raise RuntimeError(f"model_data.h 里找不到 {name}")
    body = m.group(1).replace("f,", ",").replace("f ", " ")
    return np.array([float(v) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", body)],
                    dtype=np.float32)


def _scalar(name):
    with open(MODEL_H, encoding="utf-8") as f:
        src = f.read()
    m = re.search(rf"#define\s+{name}\s+\(?\s*(-?[\d.eE+-]+)", src)
    if not m:
        raise RuntimeError(f"model_data.h 里找不到宏 {name}")
    return float(m.group(1))


def _labels():
    with open(MODEL_H, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"HG_MODEL_LABELS\[[^\]]*\]\s*=\s*\{(.*?)\}", src, re.S)
    return re.findall(r'"([^"]+)"', m.group(1))


# ---------------------------------------------------------------
# 音频读取
# ---------------------------------------------------------------

def load_wav(path, target_sr=mfcc_ref.SR):
    """读 WAV 并统一到 16kHz 单声道，返回 float PCM(-1~1)。"""
    import wave

    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if width != 2:
            raise RuntimeError(f"只支持 16bit PCM，该文件为 {width * 8}bit")
        raw = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32)

    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)

    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), target_sr)
            raw = resample_poly(raw, target_sr // g, sr // g).astype(np.float32)
        except ImportError:
            # 没有 scipy 就退化为线性插值，重采样后的高频会不准确，仅够粗看
            idx = np.arange(0, raw.size, sr / target_sr)
            raw = np.interp(idx, np.arange(raw.size), raw).astype(np.float32)
        raw = raw / 32768.0
    else:
        raw = raw / 32768.0

    return raw


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="待验证的 wav 文件")
    ap.add_argument("--threshold", type=float, default=None,
                    help="置信度阈值，默认取 config.h 的 HG_CONF_THRESHOLD")
    ap.add_argument("--confirm", type=int, default=None,
                    help="连续确认帧数，默认取 config.h 的 HG_CONFIRM_FRAMES")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args()

    if args.threshold is None:
        with open(os.path.join(ROOT, "firmware", "HomeGuard", "config.h"), encoding="utf-8") as f:
            m = re.search(r"#define\s+HG_CONF_THRESHOLD\s+([\d.]+)f", f.read())
            args.threshold = float(m.group(1))
    if args.confirm is None:
        with open(os.path.join(ROOT, "firmware", "HomeGuard", "config.h"), encoding="utf-8") as f:
            m = re.search(r"#define\s+HG_CONFIRM_FRAMES\s+(\d+)", f.read())
            args.confirm = int(m.group(1))

    mean, std = _nums("HG_NORM_MEAN"), _nums("HG_NORM_STD")
    labels = _labels()
    in_scale, in_zp = _scalar("HG_MODEL_IN_SCALE"), int(_scalar("HG_MODEL_IN_ZERO_POINT"))
    out_scale, out_zp = _scalar("HG_MODEL_OUT_SCALE"), int(_scalar("HG_MODEL_OUT_ZERO_POINT"))

    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=TFLITE)
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
        interp = Interpreter(model_path=TFLITE)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    # 量化/反量化定义与固件 quant.h 保持一致
    def quantize(x):
        return np.clip(np.round(x / in_scale) + in_zp, -128, 127).astype(np.int8)

    def dequantize(q):
        return (q.astype(np.float32) - out_zp) * out_scale

    pcm = load_wav(args.wav)
    n = mfcc_ref.WINDOW_SAMPLES
    hop = mfcc_ref.HOP

    if pcm.size < n:
        print(f"音频太短：{pcm.size} 样点，至少需要 {n}（约 {n / mfcc_ref.SR:.2f} 秒）")
        return 1

    results = []
    # 与设备端一致：每 hop(512) 出一个窗口，而不是 hop 的整数倍才判一次
    for start in range(0, pcm.size - n + 1, hop):
        seg = (pcm[start:start + n] * 32768.0).astype(np.int16)
        feat = mfcc_ref.compute_mfcc(seg)                      # (32, 40)
        x = quantize((feat - mean) / std)[None, ..., None]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        prob = dequantize(interp.get_tensor(out["index"])[0])
        cls = int(np.argmax(prob))
        results.append({
            "t": round(start / mfcc_ref.SR, 3),
            "label": labels[cls],
            "conf": float(prob[cls]),
            "probs": {labels[i]: float(prob[i]) for i in range(len(labels))},
        })

    # 决策同样复刻 decision.c：同类连续 confirm 次且都过阈值才报
    fired, streak, last = [], 0, None
    for r in results:
        if r["conf"] >= args.threshold and r["label"] == last:
            streak += 1
        elif r["conf"] >= args.threshold:
            last, streak = r["label"], 1
        else:
            last, streak = None, 0
        if streak == args.confirm:
            fired.append({"t": r["t"], "label": r["label"], "conf": r["conf"]})

    if args.json:
        print(json.dumps({"windows": results, "fired": fired}, ensure_ascii=False))
        return 0

    print(f"文件 : {os.path.basename(args.wav)}")
    print(f"长度 : {pcm.size / mfcc_ref.SR:.2f} s -> {len(results)} 个推理窗口")
    print(f"决策 : 阈值 {args.threshold:.2f}，连续 {args.confirm} 次同类确认\n")

    for r in results:
        bar = "  ".join(f"{k}={v:.2f}" for k, v in r["probs"].items())
        mark = "  <== 报警" if any(abs(f["t"] - r["t"]) < 1e-6 for f in fired) else ""
        print(f"  t={r['t']:6.2f}s  {r['label']:<12} {r['conf'] * 100:5.1f}%   {bar}{mark}")

    print()
    if fired:
        print(f"触发 {len(fired)} 次报警：")
        for f in fired:
            print(f"  t={f['t']}s  {f['label']}  {f['conf'] * 100:.1f}%")
    else:
        print("未触发报警。若设备端也没报，说明模型对该样本不敏感；"
              "若设备端报了而这里没报，优先怀疑 I2S 采样或固件特征提取。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
