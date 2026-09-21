/*
 * PC 端自测：报警决策逻辑 + INT8 量化/反量化。
 * 这两块都是纯逻辑，不依赖任何硬件，可以在 PC 上完整验证。
 */

#include <stdio.h>
#include <math.h>

#include "decision.h"
#include "quant.h"

static int failures = 0;

#define CHECK(cond, msg) do {                                  \
    if (!(cond)) { printf("  [FAIL] %s\n", msg); failures++; }  \
    else         { printf("  [ ok ] %s\n", msg); }              \
} while (0)

/* ---------------- 决策：阈值 0.85 + 同一事件连续 3 次确认 ---------------- */
enum { CLS_BABY = 0, CLS_GLASS = 1, CLS_KNOCK = 2 };

static void test_decider(void)
{
    hg_decider_t d;

    printf("阈值 0.85 + 同一事件连续三次确认\n");

    hg_decider_init(&d, 0.85f, 3);
    CHECK(hg_decider_push(&d, CLS_GLASS, 0.90f) == 0, "第1次 玻璃碎裂 0.90 -> 不报警");
    CHECK(hg_decider_push(&d, CLS_GLASS, 0.91f) == 0, "第2次同类连续 -> 仍不报警");
    CHECK(hg_decider_push(&d, CLS_GLASS, 0.92f) == 1, "第3次同类连续 -> 触发报警");

    printf("持续超阈值不应重复报警\n");
    CHECK(hg_decider_push(&d, CLS_GLASS, 0.95f) == 0, "第4次仍超阈值 -> 不重复报警");
    CHECK(hg_decider_push(&d, CLS_GLASS, 0.99f) == 0, "第5次仍超阈值 -> 不重复报警");

    printf("中途掉回阈值以下应重新计数\n");
    hg_decider_init(&d, 0.85f, 3);
    CHECK(hg_decider_push(&d, CLS_BABY, 0.90f) == 0, "第1次 0.90");
    CHECK(hg_decider_push(&d, CLS_BABY, 0.50f) == 0, "第2次 0.50 低于阈值 -> 计数清零");
    CHECK(hg_decider_push(&d, CLS_BABY, 0.90f) == 0, "第3次 0.90 -> 重新计数 1");
    CHECK(hg_decider_push(&d, CLS_BABY, 0.90f) == 0, "第4次 0.90 -> 计数 2");
    CHECK(hg_decider_push(&d, CLS_BABY, 0.90f) == 1, "第5次 0.90 -> 凑满三次，报警");

    printf("不同类别不得互相凑数（这是分类别计数的主要目的）\n");
    hg_decider_init(&d, 0.85f, 3);
    CHECK(hg_decider_push(&d, CLS_BABY, 0.90f) == 0, "婴儿哭声 0.90 -> 计数 1");
    CHECK(hg_decider_push(&d, CLS_GLASS, 0.90f) == 0, "换成玻璃碎裂 -> 重新计数 1，不报警");
    CHECK(hg_decider_push(&d, CLS_KNOCK, 0.90f) == 0, "换成敲击 -> 重新计数 1，不报警");
    CHECK(hg_decider_push(&d, CLS_KNOCK, 0.90f) == 0, "敲击连续 2 次 -> 仍不报警");
    CHECK(hg_decider_push(&d, CLS_KNOCK, 0.90f) == 1, "敲击连续 3 次 -> 报警");

    printf("边界\n");
    hg_decider_init(&d, 0.85f, 3);
    CHECK(hg_decider_push(&d, CLS_BABY, 0.84f) == 0, "0.84 恰好低于阈值 -> 不计数");
    CHECK(hg_decider_push(&d, CLS_BABY, 0.85f) == 0, "0.85 恰好等于阈值 -> 计入（含边界）");
    CHECK(hg_decider_push(&d, CLS_BABY, 0.85f) == 0, "再计一次");
    CHECK(hg_decider_push(&d, CLS_BABY, 0.85f) == 1, "凑满三次 -> 报警");

    printf("事件结束后再次触发\n");
    hg_decider_init(&d, 0.85f, 3);
    for (int i = 0; i < 3; i++) hg_decider_push(&d, CLS_GLASS, 0.90f);
    CHECK(hg_decider_push(&d, CLS_GLASS, 0.10f) == 0, "事件结束，置信度回落 -> 复位");
    for (int i = 0; i < 2; i++) hg_decider_push(&d, CLS_GLASS, 0.90f);
    CHECK(hg_decider_push(&d, CLS_GLASS, 0.90f) == 1, "同一事件二次发生 -> 可再次报警");
}

/* ---------------- INT8 量化 ---------------- */
static void test_quant(void)
{
    /* 与 model_data.h 中导出的实际常量一致 */
    const float in_scale  = 0.06698843f;
    const int   in_zp     = 7;
    const float out_scale = 0.00390625f;
    const int   out_zp    = -128;

    printf("\nINT8 量化\n");

    CHECK(hg_quantize_f32(0.0f, in_scale, in_zp) == 7, "x=0 映射到 zero_point=7");

    int8_t qmax = hg_quantize_f32(1e9f, in_scale, in_zp);
    int8_t qmin = hg_quantize_f32(-1e9f, in_scale, in_zp);
    CHECK(qmax == 127 && qmin == -128, "超出范围被饱和到 [-128, 127]");

    /* 量化 -> 反量化 的往返误差不应超过半个量化步长。
     * 注意取值范围要限定在 int8 可表示区间内，
     * 超出部分本来就会被饱和，不属于舍入误差。 */
    float lo = ((float)(-128) - (float)in_zp) * in_scale;
    float hi = ((float)(127) - (float)in_zp) * in_scale;
    float worst = 0.0f;
    for (int i = 0; i <= 2000; i++) {
        float x = lo + (hi - lo) * (float)i / 2000.0f;
        int8_t q = hg_quantize_f32(x, in_scale, in_zp);
        float rt = hg_dequantize_i8(q, in_scale, in_zp);
        float err = fabsf(rt - x);
        if (err > worst) worst = err;
    }
    printf("  (可表示区间 [%.3f, %.3f]，最大往返误差 %.6f，半量化步长 %.6f)\n",
           lo, hi, worst, in_scale * 0.5f);
    CHECK(worst <= in_scale * 0.5f + 1e-6f, "量化往返误差不超过半个量化步长");

    /* 区间外必须饱和而不是溢出回绕 */
    CHECK(hg_quantize_f32(hi * 3.0f, in_scale, in_zp) == 127, "上界外饱和到 127");
    CHECK(hg_quantize_f32(lo * 3.0f, in_scale, in_zp) == -128, "下界外饱和到 -128");

    /* softmax 输出反量化：q=-128 -> 0.0, q=127 -> 约 0.996 */
    CHECK(fabsf(hg_dequantize_i8(-128, out_scale, out_zp) - 0.0f) < 1e-6f,
          "输出 q=-128 反量化为 0.0");
    CHECK(hg_dequantize_i8(127, out_scale, out_zp) > 0.99f,
          "输出 q=127 反量化为约 1.0");
}

int main(void)
{
    printf("=== HomeGuard 决策与量化自测 ===\n\n");
    test_decider();
    test_quant();

    printf("\n");
    if (failures) {
        printf("结果：%d 项失败\n", failures);
        return 1;
    }
    printf("结果：全部通过\n");
    return 0;
}
