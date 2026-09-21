#include "wifi_driver.h"
#include "config.h"

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <HTTPClient.h>
#include <time.h>

#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

/* =====================================================
 * 设计约束
 * =====================================================
 * 音频采集任务由 I2S DMA 环形缓冲兜底，容量只有 I2S_RING_MS（约 256ms）。
 * 任何一次同步网络操作（连接 / 发布 / HTTP 请求）都可能耗时数百毫秒到数秒，
 * 一旦在主循环或采集任务里同步执行，环形缓冲就会被覆盖，
 * 表现为特征错位、置信度跳变、偶发误报。
 *
 * 因此本文件全部改成非阻塞结构：
 *   1. mqtt_send_alarm() 只做「入队」，永不触碰网络，可安全在推理路径里调用；
 *   2. 真正的 MQTT 收发放到 mqtt_loop() 里按状态机推进，每次最多发一条；
 *   3. 重连使用带上限的指数退避，绝不 while 死等；
 *   4. 触发 GitHub Actions（最慢，可能数秒）单独放到 core 0 的低优先级任务，
 *      通过二值信号量通知，主循环不感知；
 *   5. WiFi 连不上时进入离线模式：本地检测与声光报警照常工作，
 *      只是不上云，不会因为等网络而卡死整台设备。
 * ===================================================== */

#define HG_ALARM_Q_LEN      8
#define HG_WIFI_TIMEOUT_MS  20000      // 开机等 WiFi 的最长时间，超时进离线模式
#define HG_BACKOFF_MIN_MS   1000
#define HG_BACKOFF_MAX_MS   30000

typedef struct {
    char     type[16];
    int      level;
    uint32_t tick;
} AlarmMsg;

static WiFiClient   espClient;
static PubSubClient mqttClient(espClient);

static QueueHandle_t     s_alarm_q   = NULL;
static SemaphoreHandle_t s_dispatch  = NULL;   // 通知 GitHub 任务
static volatile int      s_online    = 0;      // WiFi 是否连上

static uint32_t s_last_try   = 0;
static uint32_t s_backoff    = HG_BACKOFF_MIN_MS;

/* -----------------------------------------------------
 * 慢操作：触发 GitHub Actions（PWA 拉取最新数据）
 * ----------------------------------------------------- */

#if defined(GITHUB_TOKEN) && defined(GITHUB_REPO)
static void github_dispatch_task(void *arg)
{
    (void)arg;

    for (;;) {
        /* 等通知，不占用 CPU */
        xSemaphoreTake(s_dispatch, portMAX_DELAY);

        if (WiFi.status() != WL_CONNECTED) {
            continue;
        }

        HTTPClient http;
        http.begin("https://api.github.com/repos/" + String(GITHUB_REPO) + "/dispatches");
        http.addHeader("Authorization", "token " + String(GITHUB_TOKEN));
        http.addHeader("Accept", "application/vnd.github+json");
        http.addHeader("Content-Type", "application/json");
        http.setTimeout(5000);

        int code = http.POST("{\"event_type\":\"fetch_data\"}");
        Serial.printf("[gh] dispatch -> %d\n", code);
        http.end();
    }
}
#endif

/* -----------------------------------------------------
 * MQTT 重连（单次尝试，失败返回 0）
 * ----------------------------------------------------- */

static int mqtt_try_connect(void)
{
    if (mqttClient.connected()) {
        return 1;
    }

    if (WiFi.status() != WL_CONNECTED) {
        return 0;
    }

    if (mqttClient.connect(ONENET_DEVICE_NAME,    // ClientID
                           ONENET_PRODUCT_ID,     // Username
                           ONENET_TOKEN)) {       // Password
        Serial.println("[mqtt] connected");
        s_backoff = HG_BACKOFF_MIN_MS;
        return 1;
    }

    Serial.printf("[mqtt] connect failed, state=%d, retry in %lu ms\n",
                  mqttClient.state(), (unsigned long)s_backoff);
    return 0;
}

/* -----------------------------------------------------
 * 发布一条报警（OneNET 物模型 property/post）
 * ----------------------------------------------------- */

static void mqtt_publish_one(const AlarmMsg *m)
{
    String topic = String("$sys/") + ONENET_PRODUCT_ID + "/" +
                   ONENET_DEVICE_NAME + "/thing/property/post";

    String msg = String("{\"id\":\"") + m->tick +
                 "\",\"version\":\"1.0\",\"params\":{"
                 "\"event\":{\"value\":\"" + m->type + "\"},"
                 "\"level\":{\"value\":" + m->level + "}}}";

    bool ok = mqttClient.publish(topic.c_str(), msg.c_str());

    Serial.println("========== MQTT Upload ==========");
    Serial.println(topic);
    Serial.println(msg);
    Serial.println(ok ? "Upload OK" : "Upload Failed");

    struct tm timeinfo;
    char tstr[20];
    if (getLocalTime(&timeinfo)) {
        strftime(tstr, sizeof(tstr), "%H:%M:%S", &timeinfo);
        Serial.print("Time: ");
        Serial.println(tstr);
    }
    Serial.println("=================================");

#if defined(GITHUB_TOKEN) && defined(GITHUB_REPO)
    if (ok && strlen(GITHUB_TOKEN) > 0) {
        xSemaphoreGive(s_dispatch);       // 只通知，不在这里发 HTTP
    }
#endif
}

/* -----------------------------------------------------
 * 对外接口
 * ----------------------------------------------------- */

void wifi_driver_init(void)
{
    Serial.println("[wifi] connecting...");

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - t0) < HG_WIFI_TIMEOUT_MS) {
        delay(200);
    }

    s_online = (WiFi.status() == WL_CONNECTED);

    if (s_online) {
        Serial.print("[wifi] connected, IP=");
        Serial.println(WiFi.localIP());

        /* 同步网络时间（北京时间），失败不影响主功能 */
        configTime(8 * 3600, 0, "ntp.aliyun.com", "ntp1.aliyun.com");
    } else {
        Serial.println("[wifi] FAILED -> offline mode (本地检测仍然工作，仅不上云)");
    }

    mqttClient.setServer("mqtts.heclouds.com", 1883);
    mqttClient.setBufferSize(1024);

    s_alarm_q  = xQueueCreate(HG_ALARM_Q_LEN, sizeof(AlarmMsg));
    s_dispatch = xSemaphoreCreateBinary();

#if defined(GITHUB_TOKEN) && defined(GITHUB_REPO)
    if (s_dispatch && strlen(GITHUB_TOKEN) > 0) {
        /* 慢任务放 core 0（core 1 专职音频 + 推理），优先级低于采集任务 */
        xTaskCreatePinnedToCore(github_dispatch_task, "gh_dispatch",
                                4096, NULL, 1, NULL, 0);
    }
#endif
}

void mqtt_send_alarm(const char *event_type, int level)
{
    if (!s_alarm_q || !event_type) {
        return;
    }

    AlarmMsg m;
    memset(&m, 0, sizeof(m));
    strncpy(m.type, event_type, sizeof(m.type) - 1);
    m.level = level;
    m.tick  = millis();

    /* 队列满就丢最旧的一条，绝不阻塞：报警宁可少一条，也不能拖慢音频链路 */
    if (xQueueSend(s_alarm_q, &m, 0) != pdTRUE) {
        AlarmMsg old;
        xQueueReceive(s_alarm_q, &old, 0);
        xQueueSend(s_alarm_q, &m, 0);
    }
}

void mqtt_loop(void)
{
    if (!s_alarm_q) {
        return;
    }

    /* 1. 维护连接：只在需要时尝试，带退避 */
    if (!mqttClient.connected()) {
        uint32_t now = millis();
        if (s_online && (now - s_last_try) >= s_backoff) {
            s_last_try = now;
            if (!mqtt_try_connect()) {
                s_backoff = (s_backoff * 2 > HG_BACKOFF_MAX_MS)
                                ? HG_BACKOFF_MAX_MS : s_backoff * 2;
            }
        }
    } else {
        mqttClient.loop();
    }

    /* 2. 每次最多发一条，避免一次循环里被网络拖住 */
    if (mqttClient.connected()) {
        AlarmMsg m;
        if (xQueueReceive(s_alarm_q, &m, 0) == pdTRUE) {
            mqtt_publish_one(&m);
        }
    }
}

int mqtt_is_connected(void)
{
    return mqttClient.connected() ? 1 : 0;
}
