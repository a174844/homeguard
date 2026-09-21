#ifndef HG_SOUND_EVENT_DETECT_H
#define HG_SOUND_EVENT_DETECT_H

#include <stdint.h>
#include "config.h"

/*
 * 端侧声音事件识别
 * ============================================================
 * 全部计算都在 ESP32 上完成，不依赖 PC：
 *
 *   I2S 样点
 *     -> 流式 MFCC（mfcc.c，40 维 x 32 帧）
 *     -> 按训练集统计量标准化
 *     -> INT8 量化后送入 TFLite Micro 模型（3 层 CNN）
 *     -> 取置信度最高的类别
 *     -> 阈值过滤 + 同类别连续 N 次确认（decision.c）
 *     -> 输出 SoundEvent
 *
 * 特征提取与训练侧共用同一套数学定义，并由 tools/sim 做逐点交叉比对；
 * 量化参数与标准化常量由 training/export_model.py 生成到 model_data.h，
 * 不手工抄写。
 */

/* 初始化：映射模型、分配 tensor arena、复位特征与决策状态。
 * 返回 0 成功，负值表示失败原因（见串口输出）。 */
int sound_event_detect_init(void);

/* 送入一个 hop（AUDIO_BUFFER_SIZE 个样点）。
 * 内部流式累积特征，每攒满一个推理窗口跑一次推理与决策。
 * 返回本次触发的事件；未触发返回 EVENT_NONE。 */
SoundEvent sound_event_detect_push_hop(const int16_t *hop);

/* 最近一次推理的最高置信度（0~1），供界面显示 */
float sound_event_detect_last_confidence(void);

/* 最近一次推理的类别名，供界面显示 */
const char *sound_event_detect_last_label(void);

/* tensor arena 实际占用字节数（由 TFLite Micro 统计） */
int sound_event_detect_arena_used(void);

#endif /* HG_SOUND_EVENT_DETECT_H */
