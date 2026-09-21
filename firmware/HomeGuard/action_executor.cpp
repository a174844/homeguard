#include "action_executor.h"
#include "screen_display.h"
#include "wifi_driver.h"

#include <Arduino.h>

/*
 * 动作执行器（非阻塞）
 * =====================================================
 * 原实现在报警后 delay(3000) 原地阻塞 3 秒。
 * 这 3 秒里主循环不跑、音频 hop 不取，I2S 环形缓冲（约 256ms）被反复覆盖，
 * 恢复后特征窗口已经错位，紧接着就是一次置信度跳变，容易连锁误报。
 *
 * 改成状态机：
 *   触发时记录事件 + 到期时刻，立刻返回；
 *   之后的每次 action_executor_run() 调用只检查是否到期并恢复现场。
 */

#define HG_ALARM_HOLD_MS   3000

static SoundEvent s_active = EVENT_NONE;
static uint32_t   s_until  = 0;

/* 事件 -> (显示名, 上报等级) */
static const char *event_name(SoundEvent e)
{
    switch (e) {
        case EVENT_GLASS_BREAK: return "Glass Break";
        case EVENT_BABY_CRY:    return "Baby Cry";
        case EVENT_KNOCK:       return "Knock Knock";
        default:                return NULL;
    }
}

static int event_level(SoundEvent e)
{
    switch (e) {
        case EVENT_GLASS_BREAK: return 3;
        case EVENT_BABY_CRY:    return 2;
        case EVENT_KNOCK:       return 1;
        default:                return 0;
    }
}

static const char *event_key(SoundEvent e)
{
    switch (e) {
        case EVENT_GLASS_BREAK: return "glass_break";
        case EVENT_BABY_CRY:    return "baby_cry";
        case EVENT_KNOCK:       return "knock";
        default:                return NULL;
    }
}

void action_executor_init(void)
{
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    s_active = EVENT_NONE;
    s_until  = 0;
}

void action_executor_run(SoundEvent event)
{
    /* 1. 新事件：触发报警，只做「立即可完成」的动作 */
    if (event != EVENT_NONE) {
        s_active = event;
        s_until  = millis() + HG_ALARM_HOLD_MS;

        screen_display_warning(event_name(event));

        if (event == EVENT_GLASS_BREAK) {
            digitalWrite(LED_PIN, HIGH);
        }

        /* 只入队，不碰网络 */
        mqtt_send_alarm(event_key(event), event_level(event));

        Serial.printf("[act] ALARM %s level=%d\n",
                      event_key(event), event_level(event));
        return;
    }

    /* 2. 到期恢复，同样不阻塞 */
    if (s_active != EVENT_NONE && (int32_t)(millis() - s_until) >= 0) {
        s_active = EVENT_NONE;
        digitalWrite(LED_PIN, LOW);
        screen_display_normal();
    }
}

int action_executor_busy(void)
{
    return (s_active != EVENT_NONE) ? 1 : 0;
}
