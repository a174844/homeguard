#include "screen_display.h"
#include "screen_driver.h"

#include <Arduino.h>
#include <string.h>

void screen_display_init(void) {
}

void screen_display_normal(void) {
    screen.fillScreen(TFT_BLACK);
    screen.setCursor(0, 0);
    screen.setTextSize(2);
    screen.println("System Running...");
}

void screen_display_warning(const char *event) {
    screen.fillScreen(TFT_RED);
    screen.setTextSize(3);
    screen.setCursor(10, 50);
    screen.print(event);
    screen.setTextSize(2);
}

void screen_display_update(void) {
    screen.setCursor(0, 180);
    screen.print(millis() / 1000);
}

/*
 * 空闲状态行。每帧都会被调用，因此必须避免无意义重绘：
 * 只有文本内容变化时才刷屏。
 */
void screen_display_idle(const char *label, float conf, int online) {
    static char  prev[48] = {0};
    char         cur[48];

    snprintf(cur, sizeof(cur), "%s %d%% %s",
             label ? label : "idle",
             (int)(conf * 100.0f + 0.5f),
             online ? "ON" : "OFF");

    if (strncmp(cur, prev, sizeof(prev)) == 0) {
        return;                       /* 内容没变，不重绘 */
    }
    strncpy(prev, cur, sizeof(prev) - 1);

    screen.fillScreen(TFT_BLACK);
    screen.setTextSize(2);
    screen.setCursor(0, 0);
    screen.println("Listening...");
    screen.setCursor(0, 60);
    screen.println(cur);
}
