#ifndef SDCARD_DRIVER_H
#define SDCARD_DRIVER_H
#include <SD.h>

void sdcard_driver_init();
void sdcard_write_file(const char *path, uint8_t *data, size_t len);

#endif