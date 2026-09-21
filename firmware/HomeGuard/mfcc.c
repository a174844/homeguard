#include <math.h>
#include <string.h>
#include "mfcc.h"
#include "fft.h"

#ifndef HG_PI
#define HG_PI 3.14159265358979323846f
#endif

#define PRE_EMPH   0.97f
#define LOG_EPS    1e-6f
#define INV_32768  3.0517578125e-5f      /* 1/32768 */

/* ------------------------------------------------------------------
 * 帧内工作缓冲。全部为文件级 static，构成可预期的固定占用，
 * 不在每帧重新分配。合计约 14.5KB。
 * ------------------------------------------------------------------ */
static float s_re[HG_FFT_SIZE];          /* 4KB：FFT 实部，同时兼作加窗后的时域帧 */
static float s_im[HG_FFT_SIZE];          /* 4KB：FFT 虚部 */
static float s_spec[HG_N_BINS];          /* 2KB：单边功率谱 */
static float s_logmel[HG_N_MEL];         /* 160B：log Mel 能量 */
static int   s_mel_bins[HG_N_MEL + 2];   /* 168B：Mel 滤波器频点边界 */
static int   s_mel_inited = 0;

static float hz_to_mel(float f)
{
    return 2595.0f * log10f(1.0f + f / 700.0f);
}

static float mel_to_hz(float m)
{
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

static void mel_filterbank_init(void)
{
    float fmin = 0.0f, fmax = (float)HG_SR / 2.0f;
    float mel_min = hz_to_mel(fmin), mel_max = hz_to_mel(fmax);
    int   m;

    if (s_mel_inited) {
        return;
    }

    /*
     * 只保存每个滤波器的三个频点边界（bin 索引），不保存 40x513 的稠密权重矩阵。
     * 三角权重在滤波时按边界现算，省掉约 80KB 常量表。
     */
    for (m = 0; m < HG_N_MEL + 2; m++) {
        float hz  = mel_to_hz(mel_min + (mel_max - mel_min) * (float)m / (float)(HG_N_MEL + 1));
        int   bin = (int)((hz * (float)HG_FFT_SIZE / (float)HG_SR) + 0.5f);

        if (bin < 0) {
            bin = 0;
        }
        if (bin > HG_N_BINS - 1) {
            bin = HG_N_BINS - 1;
        }
        s_mel_bins[m] = bin;
    }
    s_mel_inited = 1;
}

/*
 * 处理一帧：win16 指向 1024 个原始样点，
 * prev_sample 是紧邻该帧之前的一个样点（用于预加重）。
 * first 为 1 表示这是整段信号的第一个样点，此时不做差分（与训练侧一致）。
 * out40 接收 40 维 MFCC。
 */
static void mfcc_process_frame(const int16_t *win16, int16_t prev_sample, int first, float *out40)
{
    float xprev = (float)prev_sample;
    int   i, m;

    /* --- 预加重 + 归一化，直接写入 FFT 实部，省掉一个 4KB 的中间数组 --- */
    for (i = 0; i < HG_FRAME_LEN; i++) {
        float xcur = (float)win16[i];

        if (i == 0 && first) {
            /* 信号首点没有前一个样点，与训练侧 np.append(x[0], ...) 对齐 */
            s_re[i] = xcur * INV_32768;
        } else {
            s_re[i] = (xcur - PRE_EMPH * xprev) * INV_32768;
        }
        s_im[i] = 0.0f;
        xprev   = xcur;
    }

    /* --- 加 Hamming 窗 --- */
    for (i = 0; i < HG_FRAME_LEN; i++) {
        s_re[i] *= 0.54f - 0.46f * cosf(2.0f * HG_PI * (float)i / (float)(HG_FRAME_LEN - 1));
    }

    /* --- FFT --- */
    hg_fft(s_re, s_im, HG_FFT_SIZE);

    /* --- 功率谱 --- */
    for (i = 0; i < HG_N_BINS; i++) {
        s_spec[i] = s_re[i] * s_re[i] + s_im[i] * s_im[i];
    }

    /* --- Mel 滤波：三角权重按边界现算 --- */
    for (m = 0; m < HG_N_MEL; m++) {
        int   left   = s_mel_bins[m];
        int   center = s_mel_bins[m + 1];
        int   right  = s_mel_bins[m + 2];
        float acc    = 0.0f;

        for (i = left; i <= right; i++) {
            float w;

            if (center == left) {
                w = (i <= center) ? 1.0f : (float)(right - i) / (float)(right - center);
            } else if (i <= center) {
                w = (float)(i - left) / (float)(center - left);
            } else {
                w = (float)(right - i) / (float)(right - center);
            }
            if (w < 0.0f) {
                w = 0.0f;
            }
            acc += s_spec[i] * w;
        }
        s_logmel[m] = logf(acc + LOG_EPS);
    }

    /* --- DCT-II -> MFCC --- */
    for (i = 0; i < HG_N_MFCC; i++) {
        float acc = 0.0f;

        for (m = 0; m < HG_N_MEL; m++) {
            acc += s_logmel[m] * cosf(HG_PI * (float)i * ((float)m + 0.5f) / (float)HG_N_MEL);
        }
        out40[i] = acc;
    }
}

void hg_mfcc_reset(hg_mfcc_t *m)
{
    memset(m->history, 0, sizeof(m->history));
    m->feed     = 0;
    m->n_filled = 0;
    m->primed   = 0;
}

int hg_mfcc_push_hop(hg_mfcc_t *m, const int16_t *hop)
{
    int16_t win[HG_FRAME_LEN];

    mel_filterbank_init();

    if (!m->primed) {
        /* 第一个 hop 只作为后续重叠窗的历史，本身还凑不出完整帧 */
        memcpy(m->history, hop, sizeof(m->history));
        m->feed   = 0;
        m->primed = 1;
        return 0;
    }

    /* 重叠窗 = 上一 hop ++ 当前 hop，正好 1024 点 */
    memcpy(win, m->history, HG_HOP * sizeof(int16_t));
    memcpy(win + HG_HOP, hop, HG_HOP * sizeof(int16_t));

    /*
     * 预加重需要「窗口起点之前的那一个样点」= 上一个 hop 的最后一个样点。
     * 注意不是 history[HG_HOP-1]：那是窗口内部第 512 点，位置弄错会让
     * 流式结果与整段计算产生个位数偏差（这个 bug 由 tools/sim 的自测抓出）。
     */
    mfcc_process_frame(win, m->feed, (m->n_filled == 0) ? 1 : 0,
                       m->feat[m->n_filled]);

    /* 当前 hop 成为下一帧的历史；本轮 history 的末点成为下一帧的 feed */
    m->feed = m->history[HG_HOP - 1];
    memcpy(m->history, hop, sizeof(m->history));

    m->n_filled++;

    if (m->n_filled >= HG_N_FRAMES) {
        /* 攒满一个推理窗口，计数归零继续滑动；
         * history 保留，因此相邻窗口之间是无缝的。 */
        m->n_filled = 0;
        return 1;
    }
    return 0;
}

void hg_mfcc_compute(const int16_t *pcm, float *out)
{
    int k, i;

    mel_filterbank_init();

    for (k = 0; k < HG_N_FRAMES; k++) {
        const int16_t *win = pcm + (size_t)k * HG_HOP;
        int16_t prev = (k == 0) ? pcm[0] : pcm[(size_t)k * HG_HOP - 1];

        mfcc_process_frame(win, prev, (k == 0) ? 1 : 0, out + (size_t)k * HG_N_MFCC);
    }
    (void)i;
}

int hg_mfcc_scratch_bytes(void)
{
    return (int)(sizeof(s_re) + sizeof(s_im) + sizeof(s_spec) +
                 sizeof(s_logmel) + sizeof(s_mel_bins));
}
