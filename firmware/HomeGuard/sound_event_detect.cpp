#include "sound_event_detect.h"

#include <Arduino.h>

/*
 * TFLite Micro 用的是 ESP32 Arduino 核心自带的 espressif__esp-tflite-micro 组件，
 * 不是第三方库 TensorFlowLite_ESP32。原因见 README「构建」一节，简单说：
 * 核心的 SDK flags 里全局定义了 -DTF_LITE_STATIC_MEMORY，第三方库在该宏下
 * 会给类加上 private 的 operator delete，与 placement new 冲突，GCC 14 直接报错。
 * 核心自带的组件是用同一组 flags 预编译好的，没有这个问题，而且是 Espressif
 * 官方维护、带 ESP-NN 加速的版本。
 */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "decision.h"
#include "mfcc.h"
#include "model_data.h"
#include "quant.h"

namespace {

hg_mfcc_t     g_mfcc;                 /* 流式特征提取状态（含 32x40 特征） */
hg_decider_t  g_decider;              /* 阈值 + 连续确认 */
float         g_last_conf = 0.0f;
int           g_last_cls  = -1;
bool          g_ready     = false;

TfLiteTensor            *g_in      = nullptr;
tflite::MicroInterpreter *g_interp = nullptr;

/* tensor arena：TFLite Micro 在这里分配输入/输出与各层中间张量。
 * 大小由 training/export_model.py 按张量存活期峰值推算后写进 model_data.h。 */
alignas(16) uint8_t g_arena[HG_MODEL_ARENA_BYTES];

/* 模型实际用到的算子。
 * 这里刻意用 MicroMutableOpResolver 而不是 AllOpsResolver：
 * 新版本 TFLM 已经不再提供 AllOpsResolver，而且只注册用到的算子能明显省 flash。
 * 清单由 train_cnn.py 里的模型结构推导（Conv2D 带 same padding + ReLU 融合、
 * MaxPool2D、GlobalAveragePooling2D 落到 MEAN、Dense+Softmax），
 * 另加 Reshape/Quantize/Dequantize/Pad/Relu 作为量化与形状调整的余量。
 * 少注册一个，运行期会报 "Didn't find op for builtin opcode"，而不是静默出错。 */
using HGOpResolver = tflite::MicroMutableOpResolver<12>;

bool build_resolver(HGOpResolver &r)
{
    bool ok = true;
    ok &= r.AddConv2D()         == kTfLiteOk;
    ok &= r.AddMaxPool2D()      == kTfLiteOk;
    ok &= r.AddMean()           == kTfLiteOk;
    ok &= r.AddFullyConnected() == kTfLiteOk;
    ok &= r.AddSoftmax()        == kTfLiteOk;
    ok &= r.AddReshape()        == kTfLiteOk;
    ok &= r.AddQuantize()       == kTfLiteOk;
    ok &= r.AddDequantize()     == kTfLiteOk;
    ok &= r.AddPad()            == kTfLiteOk;
    ok &= r.AddRelu()           == kTfLiteOk;
    return ok;
}

}  // namespace

int sound_event_detect_init(void)
{
    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[infer] 模型 schema 版本不匹配: %d != %d\n",
                      (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
        return -1;
    }

    static HGOpResolver resolver;
    if (!build_resolver(resolver)) {
        Serial.println("[infer] 算子注册失败");
        return -2;
    }

    /* 用静态局部对象而不是 placement new。
     * TFLM 的类在 TF_LITE_STATIC_MEMORY 下带 private 的 operator delete，
     * 任何 new 表达式（包括 placement new 的异常清理路径）都会因访问它而编译失败。
     * 静态对象不需要调用 operator delete，也就绕开了这个限制。 */
    static tflite::MicroInterpreter interp(model, resolver, g_arena, sizeof(g_arena));
    g_interp = &interp;

    if (g_interp->AllocateTensors() != kTfLiteOk) {
        Serial.println("[infer] AllocateTensors 失败：arena 可能偏小");
        return -3;
    }

    g_in = g_interp->input(0);
    if (g_in->type != kTfLiteInt8 ||
        g_in->dims->size != 4 ||
        g_in->dims->data[1] != HG_MODEL_N_FRAMES ||
        g_in->dims->data[2] != HG_MODEL_N_MFCC) {
        Serial.println("[infer] 输入张量形状或类型与预期不符");
        return -4;
    }

    TfLiteTensor *out = g_interp->output(0);
    if (out->type != kTfLiteInt8 || out->dims->data[out->dims->size - 1] != HG_MODEL_N_CLASSES) {
        Serial.println("[infer] 输出张量与预期不符");
        return -5;
    }

    hg_mfcc_reset(&g_mfcc);
    hg_decider_init(&g_decider, HG_CONF_THRESHOLD, HG_CONFIRM_FRAMES);
    g_ready = true;

    Serial.printf("[infer] 端侧推理就绪: %d 帧 x %d 维 -> %d 类\n",
                  HG_N_FRAMES, HG_N_MFCC, HG_MODEL_N_CLASSES);
    Serial.printf("[infer] 模型 %u B(flash) | arena 配置 %d B, 实际占用 %d B\n",
                  g_model_data_len, (int)sizeof(g_arena),
                  (int)g_interp->arena_used_bytes());
    Serial.printf("[infer] 决策: 阈值 %.2f, 连续 %d 次同类确认\n",
                  (double)HG_CONF_THRESHOLD, HG_CONFIRM_FRAMES);
    return 0;
}

SoundEvent sound_event_detect_push_hop(const int16_t *hop)
{
    if (!g_ready) {
        return EVENT_NONE;
    }

    /* 还没攒满一个推理窗口，继续累积 */
    if (!hg_mfcc_push_hop(&g_mfcc, hop)) {
        return EVENT_NONE;
    }

    /* ---- 标准化 + INT8 量化，直接写进模型输入张量 ---- */
    int8_t *in = g_in->data.int8;
    for (int f = 0; f < HG_N_FRAMES; f++) {
        const float *src = g_mfcc.feat[f];
        int8_t      *dst = in + (size_t)f * HG_N_MFCC;

        for (int c = 0; c < HG_N_MFCC; c++) {
            float v = (src[c] - HG_NORM_MEAN[c]) / HG_NORM_STD[c];
            dst[c] = hg_quantize_f32(v, HG_MODEL_IN_SCALE, HG_MODEL_IN_ZERO_POINT);
        }
    }

    /* ---- 推理 ---- */
    if (g_interp->Invoke() != kTfLiteOk) {
        Serial.println("[infer] Invoke 失败");
        return EVENT_NONE;
    }

    /* ---- 反量化输出并取最大 ---- */
    const TfLiteTensor *out = g_interp->output(0);
    const int8_t       *q   = out->data.int8;

    int   best   = 0;
    float best_p = -1.0f;
    for (int i = 0; i < HG_MODEL_N_CLASSES; i++) {
        float p = hg_dequantize_i8(q[i], HG_MODEL_OUT_SCALE, HG_MODEL_OUT_ZERO_POINT);
        if (p > best_p) {
            best_p = p;
            best   = i;
        }
    }
    g_last_conf = best_p;
    g_last_cls  = best;

    /* ---- 阈值 + 同类别连续确认 ---- */
    if (!hg_decider_push(&g_decider, best, best_p)) {
        return EVENT_NONE;
    }

    Serial.printf("[infer] 触发报警: %s (置信度 %.1f%%, 连续 %d 次确认)\n",
                  HG_MODEL_LABELS[best], best_p * 100.0f, HG_CONFIRM_FRAMES);
    return (SoundEvent)HG_MODEL_CLASS_EVENT[best];
}

float sound_event_detect_last_confidence(void)
{
    return g_last_conf;
}

const char *sound_event_detect_last_label(void)
{
    return (g_last_cls >= 0 && g_last_cls < HG_MODEL_N_CLASSES)
               ? HG_MODEL_LABELS[g_last_cls]
               : "-";
}

int sound_event_detect_arena_used(void)
{
    return (g_interp != nullptr) ? (int)g_interp->arena_used_bytes() : 0;
}
