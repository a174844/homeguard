/*
 * PC 端自测：把测试 PCM 喂给端侧 MFCC 实现，导出数值供 numpy 交叉比对，
 * 同时验证「流式接口」与「整段接口」结果一致。
 *
 * 这两个检查各有意义：
 *   1) 与 numpy 比对 —— 证明 C 实现没有算错
 *   2) 流式 vs 整段   —— 证明设备上真正跑的流式路径和离线训练用的特征一致
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mfcc.h"

static int16_t pcm[HG_WINDOW_SAMPLES];
static float   feat_batch[HG_N_FRAMES * HG_N_MFCC];
static float   feat_stream[HG_N_FRAMES * HG_N_MFCC];

int main(int argc, char **argv)
{
    const char *in_path  = (argc > 1) ? argv[1] : "test_pcm.bin";
    const char *out_path = (argc > 2) ? argv[2] : "mfcc_out.txt";

    FILE *fp = fopen(in_path, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开 %s（先运行 gen_test_pcm.py 生成）\n", in_path);
        return 2;
    }
    size_t got = fread(pcm, sizeof(int16_t), HG_WINDOW_SAMPLES, fp);
    fclose(fp);
    if (got < (size_t)HG_WINDOW_SAMPLES) {
        fprintf(stderr, "输入样点不足：需要 %d，实际 %zu\n", HG_WINDOW_SAMPLES, got);
        return 2;
    }

    printf("=== HomeGuard 端侧 MFCC 自测 ===\n\n");
    printf("参数: SR=%d  FFT=%d  帧长=%d  帧移=%d  帧数=%d  Mel=%d  MFCC=%d\n",
           HG_SR, HG_FFT_SIZE, HG_FRAME_LEN, HG_HOP, HG_N_FRAMES, HG_N_MEL, HG_N_MFCC);
    printf("推理窗口: %d 样点 = %.3f s\n\n",
           HG_WINDOW_SAMPLES, (double)HG_WINDOW_SAMPLES / HG_SR);

    /* ---------- 1) 整段接口 ---------- */
    hg_mfcc_compute(pcm, feat_batch);

    int finite = 1;
    for (int i = 0; i < HG_N_FRAMES * HG_N_MFCC; i++) {
        if (!isfinite(feat_batch[i])) { finite = 0; break; }
    }
    printf("整段接口: 输出 %d 帧 x %d 维, 有限值检查 %s\n",
           HG_N_FRAMES, HG_N_MFCC, finite ? "通过" : "失败");
    printf("  C0 逐帧: ");
    for (int k = 0; k < 6; k++) {
        printf("%.2f ", feat_batch[k * HG_N_MFCC]);
    }
    printf("...\n");

    /* ---------- 2) 流式接口 ---------- */
    hg_mfcc_t st;
    int ready = 0;
    int hops  = 0;

    hg_mfcc_reset(&st);
    for (int i = 0; i * HG_HOP + HG_HOP <= HG_WINDOW_SAMPLES; i++) {
        ready = hg_mfcc_push_hop(&st, pcm + i * HG_HOP);
        hops++;
    }
    /* 流式路径产出的是最后一次攒满的结果 */
    printf("\n流式接口: 送入 %d 个 hop (%d 样点), 攒满标志=%d\n",
           hops, hops * HG_HOP, ready);

    memcpy(feat_stream, st.feat, sizeof(feat_stream));

    /* 流式与整段在数值上应当完全一致 */
    float max_stream_diff = 0.0f;
    for (int i = 0; i < HG_N_FRAMES * HG_N_MFCC; i++) {
        float d = fabsf(feat_batch[i] - feat_stream[i]);
        if (d > max_stream_diff) max_stream_diff = d;
    }
    printf("流式 vs 整段: 最大绝对偏差 = %.3e  -> %s\n",
           max_stream_diff,
           max_stream_diff == 0.0f ? "完全一致" : "存在差异，需检查");

    /* ---------- 3) 内存占用 ---------- */
    printf("\n--- 静态内存占用（由 sizeof 实测，非估算） ---\n");
    printf("  hg_mfcc_t 实例 (32x40 特征 + 512 点历史) : %6zu B\n", sizeof(hg_mfcc_t));
    printf("  帧内工作缓冲 (static)                     : %6d B\n", hg_mfcc_scratch_bytes());
    printf("  特征提取部分合计                          : %6zu B (%.1f KB)\n",
           sizeof(hg_mfcc_t) + (size_t)hg_mfcc_scratch_bytes(),
           (sizeof(hg_mfcc_t) + (size_t)hg_mfcc_scratch_bytes()) / 1024.0);

    /* ---------- 4) 导出供 numpy 比对 ---------- */
    fp = fopen(out_path, "w");
    if (!fp) {
        fprintf(stderr, "无法写出 %s\n", out_path);
        return 2;
    }
    for (int i = 0; i < HG_N_FRAMES * HG_N_MFCC; i++) {
        fprintf(fp, "%.6f\n", feat_batch[i]);
    }
    fclose(fp);
    printf("\n已导出 %s (%d 个数值)\n", out_path, HG_N_FRAMES * HG_N_MFCC);

    int bad = 0;
    if (!finite || !ready || max_stream_diff != 0.0f) {
        bad = 1;
    }
    printf("\n%s\n", bad ? "结果：存在失败项" : "结果：全部通过");
    return bad;
}
