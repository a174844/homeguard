#include <math.h>
#include "fft.h"

#ifndef HG_PI
#define HG_PI 3.14159265358979323846f
#endif

void hg_fft(float *re, float *im, int n)
{
    int i, j, k, len;

    /* 位反转置换 */
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }

    /* 蝶形运算，len 逐级翻倍 */
    for (len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * HG_PI / (float)len;
        float wr  = cosf(ang);
        float wi  = sinf(ang);
        int   half = len >> 1;

        for (i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (k = 0; k < half; k++) {
                float ur = re[i + k];
                float ui = im[i + k];
                float vr = re[i + k + half] * cr - im[i + k + half] * ci;
                float vi = re[i + k + half] * ci + im[i + k + half] * cr;

                re[i + k]        = ur + vr;
                im[i + k]        = ui + vi;
                re[i + k + half] = ur - vr;
                im[i + k + half] = ui - vi;

                /* 递推更新旋转因子，避免每个点都调用 cos/sin */
                float ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}
