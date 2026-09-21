#ifndef SCREEN_DRIVER_H
#define SCREEN_DRIVER_H

#include <LovyanGFX.hpp>
#include "config.h"

class LGFX : public lgfx::LGFX_Device
{
    lgfx::Panel_ST7735S _panel_instance;
    lgfx::Bus_SPI _bus_instance;

public:
    LGFX(void)
    {
        {
            auto cfg = _bus_instance.config();
            cfg.spi_host = SPI2_HOST;
            cfg.spi_mode = 0;
            cfg.freq_write = 40000000;
            cfg.freq_read  = 16000000;
            cfg.spi_3wire  = true;
            cfg.use_lock   = true;
            cfg.dma_channel = SPI_DMA_CH_AUTO;
            cfg.pin_sclk = SCREEN_SCK;
            cfg.pin_mosi = SCREEN_SDA;
            cfg.pin_miso = -1;
            cfg.pin_dc   = SCREEN_A0;
            _bus_instance.config(cfg);
            _panel_instance.setBus(&_bus_instance);
        }

        {
            auto cfg = _panel_instance.config();
            cfg.pin_cs   = SCREEN_CS;
            cfg.pin_rst  = SCREEN_RESET;
            cfg.pin_busy = -1;
            cfg.memory_width  = 128;
            cfg.memory_height = 160;
            cfg.panel_width   = 128;
            cfg.panel_height  = 160;
            cfg.offset_x = 0;
            cfg.offset_y = 0;
            cfg.offset_rotation = 0;
            cfg.invert   = true;
            cfg.rgb_order= false;
            cfg.dlen_16bit = false;
            cfg.bus_shared = false;
            _panel_instance.config(cfg);
        }

        setPanel(&_panel_instance);
    }
};

extern LGFX screen;
void screen_driver_init();

#endif