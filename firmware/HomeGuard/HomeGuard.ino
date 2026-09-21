#include "config.h"

#include <Arduino.h>

#include "action_executor.h"
#include "audio_capture.h"
#include "mic_driver.h"
#include "screen_display.h"
#include "screen_driver.h"
#include "sdcard_driver.h"
#include "sound_event_detect.h"
#include "wifi_driver.h"

/*
 * HomeGuard 主程序
 * ============================================================
 * 数据通路（全部在 ESP32-S3 上完成，不依赖 PC）：
 *
 *   麦克风 (I2S, 16kHz)
 *     -> audio_capture：独立任务 + 双缓冲，每次交出一个 hop(512 样点 = 32ms)
 *     -> sound_event_detect：流式 MFCC(32帧 x 40维) -> INT8 量化
 *                            -> TFLite Micro CNN 推理
 *                            -> 阈值过滤 + 同类连续确认
 *     -> action_executor：屏幕/LED 本地报警，同时把事件入队
 *     -> wifi_driver：后台把队列里的报警发到 OneNET
 *
 * 主循环里没有任何阻塞调用：不 delay 等网络、不等报警保持结束。
 * 唯一的等待是 audio_capture_take_hop() 的超时，它是采样节拍本身。
 */

static uint32_t s_last_stat = 0;
static uint32_t s_hop_count = 0;

void setup()
{
    Serial.begin(115200);
    delay(1000);
    Serial.println();
    Serial.println("================ HomeGuard ================");

    screen_driver_init();
    screen_display_init();
    screen_display_normal();

    sdcard_driver_init();
    mic_driver_init();
    audio_capture_init();
    action_executor_init();

    /* 推理就绪检查放在最前面：模型或 arena 有问题要立刻暴露，
     * 而不是等运行期出现莫名其妙的置信度。 */
    if (sound_event_detect_init() != 0) {
        Serial.println("[sys] 推理初始化失败，系统停止");
        screen_display_warning("INFER FAIL");
        for (;;) {
            delay(1000);
        }
    }

    /* WiFi / MQTT 放在最后：连不上也不影响本地检测（离线模式） */
    wifi_driver_init();

    Serial.printf("[sys] I2S 环形缓冲可覆盖 %d ms，单次推理必须小于该值\n",
                  audio_capture_ring_ms());
    Serial.println("[sys] 启动完成，开始监听");
}

void loop()
{
    /* 1. 取一个 hop。超时 100ms，正常情况下 32ms 就会来一个 */
    const int16_t *hop = audio_capture_take_hop(100);
    if (hop == NULL) {
        mqtt_loop();
        return;
    }

    /* 2. 流式特征提取 + 端侧推理 + 决策。
     *    只有在攒满一个推理窗口（33 个 hop）时才会真正跑一次 CNN。 */
    SoundEvent event = sound_event_detect_push_hop(hop);
    audio_capture_release_hop(hop);
    s_hop_count++;

    /* 3. 报警（非阻塞）或恢复现场 */
    action_executor_run(event);

    /* 4. 后台推进 MQTT：重连、心跳、发送队列里的报警 */
    mqtt_loop();

    /* 5. 空闲时刷新状态行；报警保持期间不覆盖红色报警画面 */
    if (!action_executor_busy()) {
        screen_display_idle(sound_event_detect_last_label(),
                            sound_event_detect_last_confidence(),
                            mqtt_is_connected());
    }

    /* 6. 每 10 秒打一次运行统计，便于观察是否丢 hop */
    if (millis() - s_last_stat >= 10000) {
        Serial.printf("[sys] hops=%lu (%.1f/s) | arena=%d B | mqtt=%s\n",
                      (unsigned long)s_hop_count,
                      s_hop_count / 10.0f,
                      sound_event_detect_arena_used(),
                      mqtt_is_connected() ? "up" : "down");
        s_last_stat = millis();
        s_hop_count = 0;
    }
}
