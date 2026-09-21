#ifndef HG_QUANT_H
#define HG_QUANT_H

#include <stdint.h>

/*
 * INT8 量化/反量化。
 *
 * 与训练侧 export_model.py 生成的量化参数配套：
 *     q = round(x / scale) + zero_point      （量化到 int8）
 *     x = (q - zero_point) * scale           （反量化回浮点）
 *
 * 单独抽成头文件是为了能在 PC 上直接测试，不必把整个固件拖进来。
 */

static inline int8_t hg_quantize_f32(float x, float scale, int zero_point)
{
    int v;

    if (scale <= 0.0f) {
        return 0;
    }
    v = (int)(x / scale + (x >= 0.0f ? 0.5f : -0.5f)) + zero_point;

    if (v < -128) {
        v = -128;
    }
    if (v > 127) {
        v = 127;
    }
    return (int8_t)v;
}

static inline float hg_dequantize_i8(int8_t q, float scale, int zero_point)
{
    return ((float)q - (float)zero_point) * scale;
}

#endif /* HG_QUANT_H */
