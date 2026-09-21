#include "sdcard_driver.h"
#include "config.h"

SPIClass sd_spi(HSPI);

void sdcard_driver_init() {
    sd_spi.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
    delay(100);
    if (!SD.begin(SD_CS, sd_spi)) {
       // Serial.println("SD Card Error");
    } else {
        Serial.println("SD Card OK");
    }
}

void sdcard_write_file(const char *path, uint8_t *data, size_t len) {
    File file = SD.open(path, FILE_APPEND);
    if (file) {
        file.write(data, len);
        file.close();
    }
}