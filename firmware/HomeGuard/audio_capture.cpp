#include "audio_capture.h"
#include "config.h"
#include "mfcc.h"

#include <Arduino.h>
#include <driver/i2s.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>

/* 采集的 hop 长度必须等于特征提取的帧移，否则帧边界对不上，
 * 设备上算出的特征与训练侧不是同一个东西。用编译期断言卡住。 */
#if AUDIO_BUFFER_SIZE != HG_HOP
#error "config.h 的 AUDIO_BUFFER_SIZE 必须等于 mfcc.h 的 HG_HOP"
#endif

/* 双缓冲：两个 hop 缓冲区，采集与推理各持一个 */
#define HOP_BUFS   2

static int16_t       s_buf[HOP_BUFS][HG_HOP];
static QueueHandle_t s_free_q  = NULL;   /* 空闲缓冲区指针队列 */
static QueueHandle_t s_ready_q = NULL;   /* 已填满缓冲区指针队列 */
static TaskHandle_t  s_task    = NULL;

static void audio_task(void *arg)
{
    (void)arg;

    for (;;) {
        int16_t *p = NULL;

        if (xQueueReceive(s_free_q, &p, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        /*
         * i2s_read 可能一次只返回一部分数据（取决于 DMA 粒度），
         * 这里必须循环读满 HG_HOP 个样点，否则 hop 长度参差不齐，
         * 重叠窗的相位就会漂移，特征随之整体错位。
         */
        size_t total = 0;
        while (total < HG_HOP * sizeof(int16_t)) {
            size_t got = 0;
            esp_err_t err = i2s_read(I2S_NUM_0,
                                     (uint8_t *)p + total,
                                     HG_HOP * sizeof(int16_t) - total,
                                     &got,
                                     portMAX_DELAY);
            if (err != ESP_OK || got == 0) {
                break;
            }
            total += got;
        }

        xQueueSend(s_ready_q, &p, portMAX_DELAY);
    }
}

void audio_capture_init(void)
{
    s_free_q  = xQueueCreate(HOP_BUFS, sizeof(int16_t *));
    s_ready_q = xQueueCreate(HOP_BUFS, sizeof(int16_t *));

    for (int i = 0; i < HOP_BUFS; i++) {
        int16_t *p = &s_buf[i][0];
        xQueueSend(s_free_q, &p, 0);
    }

    /* 采集任务优先级高于推理任务：优先保证数据被及时取走 */
    xTaskCreatePinnedToCore(audio_task, "hg_audio", 3072, NULL, 6, &s_task, 1);
}

const int16_t *audio_capture_take_hop(int timeout_ms)
{
    int16_t *p = NULL;

    if (s_ready_q == NULL) {
        return NULL;
    }
    if (xQueueReceive(s_ready_q, &p, pdMS_TO_TICKS(timeout_ms)) != pdTRUE) {
        return NULL;
    }
    return p;
}

void audio_capture_release_hop(const int16_t *hop)
{
    int16_t *p;

    if (hop == NULL || s_free_q == NULL) {
        return;
    }
    p = (int16_t *)hop;
    xQueueSend(s_free_q, &p, 0);
}

int audio_capture_ring_ms(void)
{
    return I2S_RING_MS;
}
