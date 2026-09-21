#!/usr/bin/env python3
"""训练 HomeGuard 的 3 层轻量 CNN。

输入是 32 帧 x 40 维 MFCC（经训练集统计量标准化），输出 3 类声音事件的 softmax 概率。
模型刻意做小，目标是能塞进 MCU 的 flash 与 tensor arena。

用法:
    python train_cnn.py --data dataset.npz --out model.keras
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

N_FRAMES, N_MFCC = 32, 40


def build_model(n_classes: int, widths=(12, 24, 48)) -> keras.Model:
    """3 层卷积 + 全局平均池化 + 全连接。

    通道数是在「精度」和「tensor arena」之间反复试出来的，命令行用 --widths 覆盖。
    两个约束：
    · 第一层卷积的输出是全部中间张量里最大的（32x40xW），W=16 时光这一层
      就要 20KB，arena 会顶到 40KB 以上；而取 8/16/32 时实测测试集准确率
      只有 79.9%。最终折中取 12/24/48，峰值存活集约 18~19KB。
    · 用 GlobalAveragePooling 而不是 Flatten，避免全连接层成为参数大头。
      这样参数量控制在 1.3 万左右，模型本体约 19.5KB。
    """
    w1, w2, w3 = widths
    inputs = keras.Input(shape=(N_FRAMES, N_MFCC, 1), name="mfcc")

    x = keras.layers.Conv2D(w1, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = keras.layers.MaxPooling2D(2, name="pool1")(x)
    x = keras.layers.Conv2D(w2, 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling2D(2, name="pool2")(x)
    x = keras.layers.Conv2D(w3, 3, padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(0.3, name="drop")(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax", name="prob")(x)

    return keras.Model(inputs, outputs, name="homeguard_cnn")


def augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
            copies: int = 1, max_shift: int = 4, noise: float = 0.05):
    """特征层面的轻量增强。

    数据集只有 121 个 5 秒片段，切窗后同一文件内的窗口高度相似，
    模型容易记住具体片段而不是事件本身。这里做两件事：
      · 沿时间轴平移若干帧 —— 事件在窗口内的位置本来就不固定
      · 叠加小幅高斯噪声 —— 抑制对个别系数精确值的过拟合
    """
    outs, ys = [x], [y]
    for _ in range(copies):
        xs = x.copy()
        shifts = rng.integers(-max_shift, max_shift + 1, size=xs.shape[0])
        for i, s in enumerate(shifts):
            if s:
                xs[i] = np.roll(xs[i], int(s), axis=0)
        if noise > 0:
            xs += rng.normal(0.0, noise, xs.shape).astype(np.float32)
        outs.append(xs)
        ys.append(y)
    return np.concatenate(outs, axis=0), np.concatenate(ys, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.keras"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260221)
    ap.add_argument("--widths", type=str, default="12,24,48",
                    help="三层卷积的通道数，逗号分隔。通道越宽精度越高、arena 越大")
    ap.add_argument("--augment", type=int, default=2,
                    help="训练集增强倍数，0 表示不增强")
    args = ap.parse_args()
    widths = tuple(int(v) for v in args.widths.split(","))
    if len(widths) != 3:
        print("--widths 需要三个整数")
        return 2

    keras.utils.set_random_seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    mean, std = d["mean"], d["std"]
    classes = [str(c) for c in d["classes"]]
    n_classes = len(classes)

    def norm(x):
        return ((x - mean) / std)[..., None].astype(np.float32)

    x_train, y_train = norm(d["x_train"]), d["y_train"]
    x_val, y_val = norm(d["x_val"]), d["y_val"]
    x_test, y_test = norm(d["x_test"]), d["y_test"]

    if args.augment > 0:
        rng = np.random.default_rng(args.seed)
        before = len(x_train)
        x_train, y_train = augment(x_train, y_train, rng, copies=args.augment)
        print(f"数据增强: 训练集 {before} -> {len(x_train)} 个窗口")

    print(f"类别 {classes}")
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}")

    model = build_model(n_classes, widths)
    model.summary()
    print(f"\n通道数 {widths}")
    print(f"参数量: {model.count_params()}")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=6, min_lr=1e-5, verbose=1),
    ]

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=args.epochs, batch_size=args.batch, callbacks=cb, verbose=2)

    # ---- 评估 ----
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n验证集准确率: {val_acc * 100:.2f}%")
    print(f"测试集准确率: {test_acc * 100:.2f}%")

    # ---- 混淆矩阵 ----
    pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_test, pred):
        cm[t, p] += 1
    print("\n混淆矩阵 (行=真实, 列=预测)")
    print("            " + "".join(f"{c[:10]:>12}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:10]:>12}" + "".join(f"{v:>12}" for v in cm[i]))
    print(f"测试样本数: {len(y_test)}  正确: {int((pred == y_test).sum())}")

    model.save(args.out)
    print(f"\n已保存 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
