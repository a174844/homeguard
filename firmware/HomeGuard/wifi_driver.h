#ifndef HG_WIFI_DRIVER_H
#define HG_WIFI_DRIVER_H

/*
 * 网络层：WiFi + OneNET MQTT + 可选的 GitHub Actions 触发
 * ============================================================
 * 这里最重要的一条约束是「绝不阻塞」。
 *
 * 音频采集任务靠 I2S 的 DMA 环形缓冲兜住 CPU 忙于推理时的样点，
 * 这个缓冲只有 config.h 里 I2S_RING_MS 那么长（约 256ms）。
 * 一旦在主循环里做同步网络请求（连接、发布、HTTP），只要耗时超过这个窗口，
 * 环形缓冲就会被写满覆盖，音频直接丢样点、特征错位。
 *
 * 所以：
 *   · mqtt_send_alarm() 只把消息塞进队列，立即返回
 *   · 实际的 MQTT 收发放在主循环里以状态机方式推进，带退避、不空转
 *   · 触发 GitHub Actions 这种慢操作单独放在低优先级任务里，用信号量通知
 */

void wifi_driver_init(void);

/* 在主循环里周期调用：推进重连、维护心跳、发送队列中的报警 */
void mqtt_loop(void);

/* 上报一次报警。仅入队，不阻塞，可安全地在任何地方调用。 */
void mqtt_send_alarm(const char *event_type, int level);

/* 当前是否已连上 MQTT（供界面显示） */
int mqtt_is_connected(void);

#endif /* HG_WIFI_DRIVER_H */
