#ifndef SCREEN_DISPLAY_H

void screen_display_init(void);
void screen_display_normal(void);
void screen_display_warning(const char *event);
void screen_display_update(void);

/* 空闲态状态行：显示当前候选类别与置信度。
 * 内部做了变化检测，只有文本真正变化时才重绘，避免高频闪烁。 */
void screen_display_idle(const char *label, float conf, int online);

#endif
