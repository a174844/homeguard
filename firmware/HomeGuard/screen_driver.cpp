#include "screen_driver.h"

LGFX screen;

void screen_driver_init() {
    pinMode(SCREEN_LED, OUTPUT);
    digitalWrite(SCREEN_LED, HIGH);
    delay(100);

    screen.init();
    screen.setRotation(0);
    screen.fillScreen(TFT_BLACK);
    screen.setTextColor(TFT_WHITE, TFT_BLACK);
    screen.setTextSize(2);
    screen.setCursor(0, 0);
    screen.println("hello");
}