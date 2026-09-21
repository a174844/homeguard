#ifndef HG_AUDIO_CAPTURE_H
#define HG_AUDIO_CAPTURE_H

#include <stdint.h>

/*
 * I2S 音频采集：双缓冲 + 独立任务
 * ============================================================
 * 采集与推理分处两个 FreeRTOS 任务，靠两个 hop 缓冲区交接：
 *
 *   采集任务：向空闲队列取一个缓冲区 -> i2s_read 填满 -> 投递到就绪队列
 *   推理任务：从就绪队列取出 -> 特征提取与推理 -> 归还到空闲队列
 *
 * 关键在于 i2s_read 实际由 DMA 完成：CPU 在算 FFT、跑 CNN 的时候，
 * I2S 外设仍在后台把样点搬进 DMA 环形缓冲，因此采集与处理是真正并行的，
 * 处理期间不会丢样点。
 *
 * 代价是有一个硬约束：单次推理耗时必须小于环形缓冲能覆盖的时长
 * （见 audio_capture_ring_ms()），否则环形缓冲会被写满覆盖、丢掉样点。
 */

void audio_capture_init(void);

/*
 * 取一个已填满的 hop。
 * 返回的指针指向内部双缓冲之一，在下一次 take 之前一直有效；
 * 用完必须调用 audio_capture_release_hop() 归还。
 * 超时返回 NULL。
 */
const int16_t *audio_capture_take_hop(int timeout_ms);

/* 归还 hop 缓冲区 */
void audio_capture_release_hop(const int16_t *hop);

/* I2S DMA 环形缓冲可覆盖的时长（毫秒） */
int audio_capture_ring_ms(void);

#endif /* HG_AUDIO_CAPTURE_H */
