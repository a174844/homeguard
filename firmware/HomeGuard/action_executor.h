#ifndef ACTION_EXECUTOR_H
#define ACTION_EXECUTOR_H

#include "config.h"

/* 初始化报警灯 GPIO */
void action_executor_init(void);

/* 在主循环里周期调用：
 *   event != EVENT_NONE -> 触发一次报警（立即返回，不阻塞）
 *   event == EVENT_NONE -> 检查保持时间是否到期并恢复现场 */
void action_executor_run(SoundEvent event);

/* 当前是否处于报警保持状态（供界面/去抖使用） */
int action_executor_busy(void);

#endif
