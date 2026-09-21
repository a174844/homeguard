#ifndef HG_FFT_H
#define HG_FFT_H

/*
 * 原地迭代 radix-2 FFT。
 * n 必须是 2 的幂。输入输出为分离的实部/虚部数组。
 *
 * 之所以自己写而不用现成库：整个特征提取链路要能在 PC 上编译、
 * 与 numpy 参考实现逐点比对，不依赖任何硬件相关的 DSP 库。
 */
void hg_fft(float *re, float *im, int n);

#endif /* HG_FFT_H */
