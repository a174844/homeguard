#!/usr/bin/env python3
"""把训练好的 Keras 模型转成 INT8 量化的 TFLite，并导出成可直接编译进固件的 C 头文件。

产物 firmware/HomeGuard/model_data.h 里除了模型数组，还带上：
  · 输入/输出的量化参数（scale / zero_point）
  · 特征标准化用的 mean / std（设备端要做同样的标准化）

这些常量如果靠手抄，迟早会抄错，所以一律由本脚本生成。

用法:
    python export_model.py --model model.keras --data dataset.npz
"""

import argparse
import os

import numpy as np
import tensorflow as tf

N_FRAMES, N_MFCC = 32, 40


def representative_dataset(x_train):
    """INT8 全量化需要一小批代表性输入来统计动态范围。"""
    def gen():
        for i in range(min(300, len(x_train))):
            yield [x_train[i:i + 1].astype(np.float32)]
    return gen


def evaluate_tflite(tflite_bytes: bytes, x: np.ndarray, y: np.ndarray):
    """用 TFLite 解释器实测量化模型的精度与量化参数。"""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    correct = 0
    for i in range(len(x)):
        sample = x[i:i + 1].astype(np.float32)
        if in_scale > 0:
            q = np.round(sample / in_scale + in_zp)
            q = np.clip(q, -128, 127).astype(np.int8)
        else:
            q = sample.astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out["index"])[0]))
        correct += int(pred == y[i])

    return correct / len(x), in_scale, in_zp, out_scale, out_zp


def c_array(data: bytes, name: str, per_line: int = 12) -> str:
    lines = [f"const unsigned char {name}[] = {{"]
    for i in range(0, len(data), per_line):
        chunk = data[i:i + per_line]
        lines.append("  " + "".join(f"0x{b:02x}, " for b in chunk).rstrip())
    lines.append("};")
    lines.append(f"const unsigned int {name}_len = {len(data)};")
    return "\n".join(lines)


def tensor_breakdown(tflite_bytes: bytes):
    """解析 flatbuffer，估算 tensor arena 的真实需求。

    注意：这个模型里中间激活张量同样带有非零的 buffer 索引
    （conv1/pool1/... 的输出都有），判据是「buffer 的数据长度为 0」而不是索引为 0，
    否则会把权重一起算进来、又把激活全部漏掉。

    arena 要装的是所有非权重张量。TFLite Micro 用贪心分配器复用生命周期
    不重叠的缓冲区，所以真实占用接近「峰值存活集」而不是「全部张量之和」。
    这里按算子的读写关系计算每个张量的存活区间，再取峰值。
    """
    from tensorflow.lite.python import schema_py_generated as schema_fb

    # TensorType -> 字节数（见 tflite schema.fbs）
    size_of = {0: 4, 1: 2, 2: 4, 3: 1, 4: 8, 6: 1, 7: 2, 8: 8, 9: 1,
               10: 8, 11: 16, 12: 8, 15: 4, 16: 2, 17: 1}

    m = schema_fb.Model.GetRootAsModel(bytearray(tflite_bytes), 0)
    sg = m.Subgraphs(0)

    sizes, names, shapes = {}, {}, {}
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        if m.Buffers(t.Buffer()).DataLength() != 0:
            continue                                    # 常量权重，不占 arena
        shape = [int(t.Shape(j)) for j in range(t.ShapeLength())]
        n = 1
        for s in shape:
            n *= max(s, 1)
        sizes[i] = n * size_of.get(int(t.Type()), 4)
        names[i] = t.Name().decode()
        shapes[i] = shape

    n_op = sg.OperatorsLength()
    first, last = {}, {}
    for j in range(n_op):
        op = sg.Operators(j)
        for k in range(op.InputsLength()):
            ti = int(op.Inputs(k))
            if ti in sizes:
                first.setdefault(ti, 0)                 # 图输入从第 0 个算子起就活着
                last[ti] = max(last.get(ti, j), j)
        for k in range(op.OutputsLength()):
            ti = int(op.Outputs(k))
            if ti in sizes:
                first[ti] = j
                last.setdefault(ti, j)                  # 图输出到最后一个算子仍在用

    peak = 0
    for j in range(n_op):
        live = sum(sizes[t] for t in sizes
                   if first.get(t, 0) <= j <= last.get(t, n_op - 1))
        peak = max(peak, live)

    detail = [(names[t], shapes[t], sizes[t]) for t in sorted(sizes)]
    return sum(sizes.values()), peak, max(sizes.values()), detail


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "model.keras"))
    ap.add_argument("--data", default=os.path.join(here, "dataset.npz"))
    ap.add_argument("--out-header", default=os.path.join(here, "..", "firmware", "HomeGuard", "model_data.h"))
    ap.add_argument("--out-tflite", default=os.path.join(here, "model_int8.tflite"))
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    mean, std = d["mean"].astype(np.float32), d["std"].astype(np.float32)
    classes = [str(c) for c in d["classes"]]

    def norm(x):
        return ((x - mean) / std)[..., None].astype(np.float32)

    x_train, x_test, y_test = norm(d["x_train"]), norm(d["x_test"]), d["y_test"]

    model = tf.keras.models.load_model(args.model)
    _, float_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"float32 模型测试集准确率: {float_acc * 100:.2f}%")
    print(f"参数量: {model.count_params()}")

    # ---- INT8 全量化 ----
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(x_train)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_bytes = conv.convert()

    with open(args.out_tflite, "wb") as f:
        f.write(tflite_bytes)
    print(f"\nINT8 模型: {len(tflite_bytes)} bytes -> {args.out_tflite}")

    int8_acc, in_scale, in_zp, out_scale, out_zp = evaluate_tflite(tflite_bytes, x_test, y_test)
    print(f"INT8 量化后测试集准确率: {int8_acc * 100:.2f}%  "
          f"(相对 float32 变化 {(int8_acc - float_acc) * 100:+.2f} 个百分点)")
    print(f"输入量化: scale={in_scale:.8f} zero_point={in_zp}")
    print(f"输出量化: scale={out_scale:.8f} zero_point={out_zp}")

    # ---- arena 需求分析 ----
    act_total, act_peak, act_biggest, acts = tensor_breakdown(tflite_bytes)
    # 除张量本身，算子还需内部临时缓冲（卷积的 im2col 等），留 4KB 余量
    arena = int((act_peak + 4096) // 1024 + 1) * 1024

    print(f"\n非权重张量合计: {act_total} B ({act_total / 1024:.1f} KB)")
    print(f"峰值存活集    : {act_peak} B ({act_peak / 1024:.1f} KB)   <- 贪心分配下的实际需求")
    print(f"最大单个张量  : {act_biggest} B ({act_biggest / 1024:.1f} KB)")
    for name, shape, b in acts:
        print(f"    {name:<58} {str(shape):<20} {b:>7} B")
    print(f"据此配置 tensor arena: {arena} B ({arena / 1024:.0f} KB) = 峰值 + 4KB 余量")

    # ---- 生成 C 头文件 ----
    def arr(nums, name, per_line=8):
        """生成 C 浮点数组。

        注意格式化模板不要带成对的 {{ }}：这里 arr() 是在下面的 f-string 里
        作为表达式调用的，而 f-string 的「{{ -> {」转义只作用于字面文本部分，
        对表达式内部的字符串不生效。写成 "{{:.6f}}f" 会被原样传进来，
        再被 .format() 还成 "{:.6f}f"，最终写进头文件的是一堆占位符而不是数字
        —— 编译期就会崩。所以模板直接写单花括号。
        """
        fmt = "{:.6f}f"
        body = []
        for i in range(0, len(nums), per_line):
            body.append("    " + ", ".join(fmt.format(v) for v in nums[i:i + per_line]) + ",")
        return f"static const float {name}[{len(nums)}] = {{\n" + "\n".join(body) + "\n};"

    mean_arr = arr(mean.tolist(), "HG_NORM_MEAN")
    std_arr = arr(std.tolist(), "HG_NORM_STD")

    labels = ", ".join(f'"{c}"' for c in classes)

    # 类别名 -> config.h 里 SoundEvent 的枚举值
    event_enum = {"glass_break": 1, "baby_cry": 2, "knock": 3}
    missing = [c for c in classes if c not in event_enum]
    if missing:
        print(f"错误：类别 {missing} 没有对应的 SoundEvent 枚举值，请先在 export_model.py 中补充映射")
        return 2
    class_event = ", ".join(f"{event_enum[c]} /*{c}*/" for c in classes)
    hdr = f"""/*
 * 本文件由 training/export_model.py 自动生成，请勿手工修改。
 *
 * 模型结构 : 3 层卷积 + 全局平均池化 + 全连接(softmax)
 * 输入     : {N_FRAMES} 帧 x {N_MFCC} 维 MFCC（已按 NORM_MEAN/NORM_STD 标准化）
 * 量化     : INT8 全量化（输入输出均为 int8）
 * float32 测试集准确率 : {float_acc * 100:.2f}%
 * INT8    测试集准确率 : {int8_acc * 100:.2f}%
 */

#ifndef HG_MODEL_DATA_H
#define HG_MODEL_DATA_H

#include <stdint.h>

#define HG_MODEL_N_FRAMES   {N_FRAMES}
#define HG_MODEL_N_MFCC     {N_MFCC}
#define HG_MODEL_N_CLASSES  {len(classes)}

/* 输入张量的量化参数：q = round(x / scale) + zero_point */
#define HG_MODEL_IN_SCALE       {in_scale:.8f}f
#define HG_MODEL_IN_ZERO_POINT  ({in_zp})

/* 输出张量的量化参数（softmax 输出）：p = (q - zero_point) * scale */
#define HG_MODEL_OUT_SCALE      {out_scale:.8f}f
#define HG_MODEL_OUT_ZERO_POINT ({out_zp})

/* tensor arena 大小。
 * 由导出脚本按「张量存活期峰值」推算：
 *   非权重张量合计 {act_total} B（不叠加，贪心分配会复用）
 *   峰值存活集     {act_peak} B（贪心分配下的实际需求）
 *   最大单个张量   {act_biggest} B
 * 配置值 {arena} B = 峰值 + 4KB 余量（卷积的 im2col 等算子需要临时缓冲）。
 * 首次烧录请核对串口打印的 interpreter->arena_used_bytes()，
 * 确认确实有余量后再调小，不要把这里的余量当成安全边界直接砍掉。 */
#define HG_MODEL_ARENA_BYTES    ({arena})

static const char *const HG_MODEL_LABELS[HG_MODEL_N_CLASSES] = {{ {labels} }};

/* ---- 模型类别序号 -> SoundEvent 枚举值（枚举定义见 config.h） ----
 * 由导出脚本按类别名自动映射，避免手工对照出错。
 * 顺序与 HG_MODEL_LABELS 一致。 */
static const int HG_MODEL_CLASS_EVENT[HG_MODEL_N_CLASSES] = {{ {class_event} }};

/* 特征标准化参数：x_norm = (x - mean) / std */
{mean_arr}

{std_arr}

/* ---- 模型本体（.tflite flatbuffer） ---- */
{c_array(tflite_bytes, "g_model_data")}

#endif /* HG_MODEL_DATA_H */
"""
    with open(args.out_header, "w", encoding="utf-8") as f:
        f.write(hdr)
    print(f"已生成 {os.path.abspath(args.out_header)}  "
          f"({os.path.getsize(args.out_header) / 1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
