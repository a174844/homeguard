#include "mic_driver.h"
#include "config.h"

#include <Arduino.h>
#include <driver/i2s.h>

void mic_driver_init(void)
{
    /*
     * dma_buf_count x dma_buf_len 决定 DMA 环形缓冲能缓存多少样点，
     * 也就是「CPU 在跑推理时最多能攒多久的音频」。
     *
     * 原始工程用 8 x 64，只有 512 个样点（16kHz 下约 32ms），
     * 而一次 CNN 推理通常要几十毫秒，缓冲必然被打满、样点被覆盖。
     * 这里改为 8 x 512 = 4096 个样点，约 256ms，给推理留出余量。
     * 具体时长由 config.h 的 I2S_RING_MS 给出。
     */
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = I2S_DMA_BUF_COUNT,
        .dma_buf_len = I2S_DMA_BUF_LEN,
        .use_apll = false
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = MIC_SCK,
        .ws_io_num = MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = MIC_SD
    };

    esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[mic] i2s_driver_install 失败: %d\n", (int)err);
        return;
    }

    err = i2s_set_pin(I2S_NUM_0, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[mic] i2s_set_pin 失败: %d\n", (int)err);
        return;
    }

    i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);

    Serial.printf("[mic] I2S 就绪: %d Hz, DMA %d x %d = %d 样点 (%d ms)\n",
                  SAMPLE_RATE, I2S_DMA_BUF_COUNT, I2S_DMA_BUF_LEN,
                  I2S_RING_SAMPLES, I2S_RING_MS);
}
