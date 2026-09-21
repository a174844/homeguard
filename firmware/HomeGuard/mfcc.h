#ifndef HG_MFCC_H
#define HG_MFCC_H

#include <stdint.h>

/* 本文件由 C 实现（mfcc.c），但会被 C++ 的 sound_event_detect.cpp 引用。
 * 没有 extern "C" 的话，C++ 会按 C++ 规则改名（name mangling），
 * 链接期报 undefined reference —— 编译能过、链接才炸，很难一眼看出原因。 */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * 端侧 MFCC 特征提取（流式）
 * ============================================================
 * 参数必须与训练侧 training/mfcc_ref.py 完全一致，
 * 否则训练出的模型在设备上会得到分布不同的特征，精度直接崩掉。
 *
 * 帧结构：帧长 1024 点(64ms)，帧移 512 点(32ms)，相邻帧重叠 50%。
 * 一次推理窗口 = 32 帧，覆盖 (32-1)*512 + 1024 = 16896 点 ≈ 1.056s。
 *
 * 流式设计的原因：若先缓存整段 16896 点再做特征，光音频缓冲就要 33KB，
 * 加上浮点中间量会超过 65KB。改为每来一个 hop 就产出一帧，
 * 只需要「上一 hop + 当前 hop」两个 512 点缓冲区就能拼出重叠窗，
 * 静态缓冲压到 20KB 出头。
 */

#define HG_SR          16000                        /* 采样率 */
#define HG_FFT_SIZE    1024                         /* FFT 点数，2 的幂 */
#define HG_FRAME_LEN   1024                         /* 帧长(样点) = 64ms */
#define HG_HOP         512                          /* 帧移(样点) = 32ms */
#define HG_N_FRAMES    32                           /* 每窗口帧数 */
#define HG_N_MEL       40                           /* Mel 滤波器个数 */
#define HG_N_MFCC      40                           /* MFCC 维数 */
#define HG_N_BINS      (HG_FFT_SIZE / 2 + 1)        /* 单边谱线数 */

/* 一次推理窗口覆盖的样点数 */
#define HG_WINDOW_SAMPLES ((HG_N_FRAMES - 1) * HG_HOP + HG_FRAME_LEN)

typedef struct {
    float    feat[HG_N_FRAMES][HG_N_MFCC];  /* 累积的 40 维 x 32 帧特征 */
    int16_t  history[HG_HOP];               /* 上一 hop，拼出重叠窗的前半段 */
    int16_t  feed;                          /* 紧邻 history 之前的最后一个样点，供预加重使用 */
    int      n_filled;                      /* 已累积帧数；同时决定该帧是否按信号起点处理 */
    uint8_t  primed;                        /* 是否已装载首个 hop */
} hg_mfcc_t;

/* 复位（保留 history 可继续滑动；cold=1 时连 history 一起清空） */
void hg_mfcc_reset(hg_mfcc_t *m);

/*
 * 送入一个 hop（HG_HOP 个样点）。
 * 返回 1 表示已攒满 HG_N_FRAMES 帧，m->feat 可以送推理；
 * 返回 0 表示还没攒满。攒满后内部计数自动归零，继续接收下一窗口。
 */
int hg_mfcc_push_hop(hg_mfcc_t *m, const int16_t *hop);

/*
 * 一次性计算整段 PCM 的特征，供 PC 端自测与交叉比对使用。
 * pcm 长度需 >= HG_WINDOW_SAMPLES；out 按 [frame][coeff] 排布，长度 32*40。
 */
void hg_mfcc_compute(const int16_t *pcm, float *out);

/*
 * 返回帧内工作缓冲的字节数（文件级 static，固定占用）。
 * 供内存占用核查使用，避免对外给出没有依据的估算。
 */
int hg_mfcc_scratch_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* HG_MFCC_H */
