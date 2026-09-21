# HomeGuard

基于 ESP32-S3 的**本地声音事件检测**系统：麦克风采集 → 端侧 MFCC → INT8 量化 CNN → 本地声光报警 + 上云。

所有音频处理与神经网络推理都在 MCU 上完成，**不上传任何音频、不依赖 PC**。网络只用于把已经判定好的事件上报给 OneNET。

移动端看板（PWA）：<https://a174844.github.io/homeguard/>

---

## 1. 它解决什么问题

家庭场景里有三类声音值得立刻响应：玻璃破碎（入侵）、婴儿哭声（看护）、敲门（访客）。
常见的做法是设备把录音传到服务器识别，代价是隐私、延迟和断网即失效。

这里换成另一条路：把模型压到能在 ESP32-S3 上跑的规模， MCU 自己听、自己判，只在判定成立时上报一条文本。

---

## 2. 系统结构

```
              ┌──────────────── I2S ────────────────┐
              │  INMP441 数字麦克风  16kHz / 16bit  │
              └───────────────────┬─────────────────┘
                                  │  DMA 环形缓冲 (8 x 512 = 256ms)
                          ┌───────▼────────┐
                    core1 │ audio_capture  │  FreeRTOS 任务，prio 6
                          │  双缓冲交出 hop │  每次 512 样点 = 32ms
                          └───────┬────────┘
                                  │ 队列
                          ┌───────▼────────────────────────────┐
                    core1 │ sound_event_detect                 │ loop 任务，prio 1
                          │  流式 MFCC 32帧 x 40维             │
                          │  标准化 -> INT8 量化               │
                          │  TFLite Micro CNN 推理             │
                          │  阈值 + 同类别连续确认             │
                          └───────┬────────────────────────────┘
                                  │ SoundEvent
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     action_executor      screen_display      wifi_driver
     屏幕 / LED 报警        OLED 状态行          OneNET MQTT 上报
                                                 （队列 + 后台任务）
```

### 为什么网络层必须做成非阻塞

这是整个工程最容易踩的一坑。

I2S 的 DMA 环形缓冲只有 **256ms**。如果在主循环里同步做一次 MQTT 连接或 HTTP 请求，耗时轻易超过这个窗口，
环形缓冲就会被写满覆盖 —— 表现是特征错位、置信度跳变、随机误报，而且很难和"模型不准"区分开。

所以：

| 原来的做法 | 现在的做法 |
| --- | --- |
| `mqtt_reconnect()` 里 `while(!connected) delay(3000)` | 带上限的指数退避，1s → 30s，失败立即返回 |
| `mqtt_send_alarm()` 里同步 publish + HTTP | 只入队，`mqtt_loop()` 每次最多发一条 |
| GitHub Actions 触发在报警路径里同步执行 | 独立任务挂 core 0，二值信号量通知 |
| WiFi 连不上就死等 | 20s 超时后进入**离线模式**，本地检测照常工作 |
| `action_executor_run()` 里 `delay(3000)` 保持报警 | 记录到期时刻的状态机，下一拍恢复 |

---

## 3. 端侧推理流水线

| 环节 | 参数 |
| --- | --- |
| 采样 | 16 kHz / 16 bit / 单声道 |
| 分帧 | 帧长 1024 点 (64ms)，帧移 512 点 (32ms)，重叠 50% |
| Mel | 40 个三角滤波器，0 ~ 8kHz |
| MFCC | 40 维，DCT-II |
| 窗口 | 32 帧 → `(32-1)×512 + 1024 = 16896` 点 ≈ **1.056 秒** |
| 标准化 | 训练集统计量的 mean / std（由导出脚本写进 `model_data.h`） |
| 量化 | INT8 全量化，`q = round(x/scale) + zero_point` |
| 模型 | 3 层 Conv2D(12/24/48) + MaxPool → GlobalAveragePool → Dense(3) + Softmax |
| 决策 | 置信度 ≥ 0.85 且**同一类别连续 3 次**才报警 |

### 流式 MFCC 的必要性

若先缓存整段 16896 点再做特征，光音频缓冲就要 33KB，加上浮点中间量会撑爆内部 RAM。

改成每来一个 hop 就产出一帧，只需要「上一 hop + 当前 hop」两个 512 点缓冲就能拼出重叠窗。
实测静态占用 **16.3 KB**（`hg_mfcc_t` 6156 B + 帧内工作缓冲 10572 B，由 `sizeof` 实算而非估算）。

---

## 4. 模型与精度

数据集为 121 个 wav（baby_cry 41 / glass_break 40 / knock 40），原始 44.1kHz 重采样到 16kHz 后按 1.056s 窗口、步长 0.25s 切片，得到 1936 个样本。

划分方式按**文件**而不是按窗口 —— 同一个音频文件切出的窗口高度相关，按窗口随机划分会造成数据泄漏、测试集准确率虚高。

| 划分 | 样本数 |
| --- | --- |
| train | 1360 |
| val | 288 |
| test | 288 |

| 模型 | 测试集准确率 |
| --- | --- |
| float32 | **85.07%** |
| INT8 量化 | **84.72%**（量化损失 −0.35 个百分点） |

资源占用：

| 项 | 数值 | 位置 |
| --- | --- | --- |
| 参数量 | 13,299 | — |
| 模型本体 | 19,976 B | flash（编译进固件） |
| tensor arena | 23,552 B 配置 | RAM（峰值存活集约 18~19 KB，再加 4KB 余量，精确值见 `export_model.py` 输出） |
| 特征提取静态缓冲 | 16,728 B | RAM |

整固件实测量（ESP32-S3，`arduino-cli compile` 输出）：

```
Sketch uses 1182387 bytes (90%) of program storage space. Maximum is 1310720 bytes.
Global variables use 90276 bytes (27%) of dynamic memory. Maximum is 327680 bytes.
```

> arena 的需求是用脚本解析 flatbuffer、按**张量存活期峰值**（而非简单求和）推算的。
> 非权重张量合计约 34 KB，但 TFLite Micro 用贪心分配复用生命周期不重叠的缓冲区，
> 实际需求远小于这个和。运行 `export_model.py` 会逐项打印每个张量的形状与字节数，
> 以及据此配置的 arena 大小；详见其中的 `tensor_breakdown()`。

### 内存口径

"峰值内存 30 KB 以内"说的是**采集 + 特征提取链路**，这里逐项对得上（全部 `sizeof` 实算）：

| 项 | 字节 | 说明 |
| --- | --- | --- |
| 采集双缓冲 `s_buf[2][512]` | 2,048 | 两个 hop 缓冲区轮流工作 |
| `hg_mfcc_t` | 6,156 | 32×40 特征累积 + 上一 hop + 状态 |
| 帧内工作缓冲 | 10,572 | FFT 实/虚部、功率谱、logMel、Mel 边界 |
| **合计** | **18,776 B ≈ 18.3 KB** | |

为什么能压到这个量级：特征提取是**流式**的，每来一个 hop 只算一帧，只需要
「上一 hop + 当前 hop」两个 512 点缓冲就能拼出 50% 重叠窗。若换成"先缓存整段
16896 点再算特征"的写法，光原始样点就要 `16896 × 2 = 33.8 KB`，已经超出 30 KB。

需要分开看的是：模型推理的 tensor arena 是**另一笔独立开销**（配置 23,552 B，
实际峰值约 18.8 KB）。两者相加约 42 KB，整机静态 RAM 实测 90,276 B，
其余主要是 WiFi / MQTT / FreeRTOS 任务栈。三个数字口径不同，不要混着引用。

---

## 5. 仓库结构

```
.
├── firmware/HomeGuard/           Arduino 固件
│   ├── HomeGuard.ino             主程序（setup/loop）
│   ├── config.h                  硬件引脚、常量、推理开关
│   ├── secrets.h.example         凭据模板（复制为 secrets.h 后填写）
│   ├── mic_driver.*              I2S 初始化
│   ├── audio_capture.*           采集任务 + 双缓冲
│   ├── fft.* / mfcc.*            端侧特征提取
│   ├── quant.h                   INT8 量化/反量化
│   ├── decision.*                阈值 + 连续确认
│   ├── model_data.h              模型与量化参数（脚本生成，勿手改）
│   ├── sound_event_detect.*      TFLite Micro 推理
│   ├── action_executor.*         声光报警状态机
│   ├── wifi_driver.*             WiFi / MQTT / Actions（全非阻塞）
│   └── screen_*.h/cpp  sdcard_*  屏幕与 SD 卡
├── training/
│   ├── prepare_dataset.py        wav → 特征数据集
│   ├── mfcc_ref.py               端侧 MFCC 的 numpy 等价实现
│   ├── train_cnn.py              训练
│   └── export_model.py           INT8 量化 + 导出 model_data.h
├── tools/
│   ├── build.sh                  命令行编译固件（arduino-cli）
│   ├── sim/                      PC 端算法自测（无需 ESP32 工具链）
│   └── pc_verify/verify_wav.py   PC 端完整跑一遍流水线，用于对照排查
├── index.html / manifest.json / sw.js   PWA 看板
└── .github/workflows/proxy.yml
```

---

## 6. 快速开始

### 6.1 固件

1. 安装 **Arduino ESP32 核心 3.x**（实测 3.3.10）
2. 安装库：**`LovyanGFX`**、**`PubSubClient`**
   （`SD.h` / `WiFi.h` / `HTTPClient.h` 都属于核心自带，不需要额外装）
3. 复制 `firmware/HomeGuard/secrets.h.example` 为 `secrets.h`，填入自己的值：

```c
#define WIFI_SSID       "your-ssid"
#define WIFI_PASSWORD   "your-password"
#define ONENET_PRODUCT_ID  "xxxx"
#define ONENET_DEVICE_NAME "xxxx"
#define ONENET_TOKEN       "xxxx"
#define GITHUB_TOKEN    ""      // 留空则完全不触发 GitHub Actions
```

> `secrets.h` 已在 `.gitignore` 中排除。**不要把真实凭据提交进仓库。**

4. 开发板选 **ESP32S3 Dev Module**，烧录。

命令行编译（不开 IDE）：

```bash
./tools/build.sh                                   # 自动查找 arduino-cli
ARDUINO_CLI=/path/to/arduino-cli ./tools/build.sh  # 或手动指定
```

> 构建目录必须是不含非 ASCII 字符的路径。如果工作副本在中文目录下，
> GNU ld 会报 `cannot open output file ... No such file or directory`，
> `build.sh` 默认把产物放到临时目录规避这一点。

#### 关于 TFLite 库：不要用第三方 TensorFlowLite_ESP32

推理用的是 **ESP32 核心自带的 `espressif__esp-tflite-micro` 组件**（官方 Espressif 版本，
带 ESP-NN 加速，核心已经自动链接 `libespressif__esp-tflite-micro.a`，不需要额外装任何东西）。

不要用第三方库 `TensorFlowLite_ESP32`，它在核心 3.x 下编不过，原因有两层：

1. 核心的 SDK flags 全局定义了 `-DTF_LITE_STATIC_MEMORY`。该宏下 TFLM 的
   `TF_LITE_REMOVE_VIRTUAL_DELETE` 会给类加一个 **private 的 `operator delete`**，
   而 `new (buf) T()` 的异常清理路径要调用它，GCC 14 直接报
   `error: assignment of read-only member ... is private within this context`。
2. 该库还打包了一堆 LCD 驱动（`src/screen/controller_driver/*.c`），
   在 GCC 14 下会因隐式函数声明（已升格为错误）批量失败。

规避方式：核心自带的组件是用同一组 flags 预编译好的，没有这些问题；
另外代码里用**静态局部对象**构造 `MicroInterpreter` 而不是 placement new，
也就不会触发那个 private `operator delete`。

串口会打印：

```
[infer] 端侧推理就绪: 32 帧 x 40 维 -> 3 类
[infer] 模型 19976 B(flash) | arena 配置 23552 B, 实际占用 xxxx B
[mic] I2S 就绪: 16000 Hz, DMA 8 x 512 = 4096 样点 (256 ms)
[sys] hops=313 (31.3/s) | arena=xxxx B | mqtt=up
```

`hops` 稳定在 **31.25/s** 说明没丢 hop（16000 / 512 = 31.25）。如果明显偏低，说明推理太慢或网络阻塞了主循环。

### 6.2 PC 端算法自测

不需要任何 ESP32 工具链，纯 C99 + libm：

```bash
cd tools/sim
make test
```

会做三件事：

1. `dump_mfcc` —— 对同一段 PCM 分别走「整段」和「流式」两条路径，验证结果**完全一致**（实测最大偏差 0.000e+00）
2. `compare_mfcc.py` —— C 实现 vs numpy 参考实现逐点比对（最大相对偏差 1.5e-06，float32 精度量级）
3. `test_decision` —— 决策逻辑与 INT8 量化往返的单元测试

这一步是「设备端特征 == 训练侧特征」的保证机制。只要训练侧和设备侧任一处参数改动而另一侧没跟上，`make test` 就会失败。

### 6.3 PC 端对照验证

设备上出现「明明有声音却不报」时，用它判断问题在哪一层：

```bash
python tools/pc_verify/verify_wav.py recording.wav
```

它用**同一个 INT8 模型**和**同一组量化/标准化参数**（直接从 `model_data.h` 解析，不从别处复制）重跑整条流水线。

- PC 能识别、设备不能 → 查麦克风与 I2S 采样链路
- 两边都不能 → 模型对该样本不敏感

---

## 7. 重新训练自己的模型

```bash
cd training
python prepare_dataset.py --data "<数据目录>" --out dataset.npz
python train_cnn.py --data dataset.npz --out model.keras
python export_model.py --model model.keras --data dataset.npz
```

数据目录按类别分子目录：

```
<数据目录>/
├── baby_cry/     *.wav
├── glass_break/  *.wav
└── knock/        *.wav
```

`export_model.py` 会同时做这几件事：

- 转 INT8 全量化，并在测试集上**实测**量化后的准确率
- 按张量存活期峰值推算 tensor arena 需求
- 把模型数组、量化参数、标准化 mean/std、类别到 `SoundEvent` 的映射一起写进 `firmware/HomeGuard/model_data.h`

**不要手改 `model_data.h`** —— 里面的常量一旦与 firmware 不同步，设备端精度会静默下降且极难排查。

> 原始音频样本为第三方素材，受其原始许可约束，不随本仓库分发。

---

## 8. 实测数据

| 项 | 结果 |
| --- | --- |
| MFCC / FFT / 决策 / 量化 | ✅ PC 端编译并跑通（`cd tools/sim && make test`）；流式 vs 整段偏差 0.000e+00，C vs numpy 相对偏差 1.5e-06 |
| 固件整体编译链接 | ✅ `arduino-cli compile --fqbn esp32:esp32:esp32s3` 干净构建 0 错误 |
| flash / RAM 占用 | ✅ 实测 1,182,387 B flash（90%）、90,276 B 静态 RAM（27%） |

### 算子注册

用的是 `MicroMutableOpResolver<12>`，注册了 10 个算子：Conv2D / MaxPool2D / Mean /
FullyConnected / Softmax / Reshape / Quantize / Dequantize / Pad / Relu。

其中 Conv2D、MaxPool2D、Mean、FullyConnected、Softmax 是模型结构直接需要的；
后 5 个是量化与形状调整可能引入的余量。少注册一个，运行期会在 `AllocateTensors()`
阶段明确报 `Didn't find op for builtin opcode`，不会静默出错，
所以上板后如果看到这条，直接往 `build_resolver()` 里补即可。

flash 已经用到 90%，如果后续要压缩体积，优先从这两处下手：
删掉确实没用到的算子注册、把 `HG_MODEL_ARENA_BYTES` 按实测值调小。

---

## 9. 许可

MIT，见 [LICENSE](./LICENSE)。
