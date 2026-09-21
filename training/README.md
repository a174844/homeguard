# 训练侧说明

```bash
python prepare_dataset.py --data "<数据目录>" --out dataset.npz
python train_cnn.py --data dataset.npz --out model.keras
python export_model.py --model model.keras --data dataset.npz
```

依赖：`numpy`、`scipy`、`tensorflow`。版本要求见 `requirements.txt`。

## 数据目录格式

```
<数据目录>/
├── baby_cry/     *.wav
├── glass_break/  *.wav
└── knock/        *.wav
```

类别名会原样写进 `model_data.h`，并需要能在 `export_model.py` 的 `event_enum` 里找到对应的
`SoundEvent` 枚举值；新增类别时必须同时补这个映射，否则导出会直接报错而不会静默生成错误的头文件。

原始音频为第三方素材，受其原始许可约束，不随本仓库分发。

## 三条容易踩的规矩

1. **特征提取只能用 `mfcc_ref.py`，不要换 librosa。**
   设备端 MFCC 的每一个参数（帧长、Mel 点数、log 底数、DCT 归一化、预加重首点处理）
   都必须和这里逐项一致。一旦不一致，训练准确率再高，设备上也是废的，而且极难排查。

2. **划分 train/val/test 按文件，不按窗口。**
   同一个音频切出的窗口高度相关，按窗口随机划分会造成数据泄漏，测试集准确率虚高。

3. **不要把这些常量手抄进固件。**
   量化参数和标准化 mean/std 一律由 `export_model.py` 写进 `model_data.h`。

## 产物

| 文件 | 是否入库 | 说明 |
| --- | --- | --- |
| `dataset.npz` | 否 | 中间产物，换数据集即失效 |
| `model.keras` | 是 | float32 模型，便于继续训练 |
| `model_int8.tflite` | 是 | 设备端实际使用的量化模型 |
| `firmware/HomeGuard/model_data.h` | 是 | 脚本生成，勿手改 |
