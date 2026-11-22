# YOLOv5 行人检测（视频作业）

自选一段视频，抽帧自建数据集，微调 YOLOv5s 检测画面中的行人，并用训练好的权重对同一段视频推理，
输出检测视频、三帧截图与权重文件。

## 1. 视频来源

- 文件：`assets/people-detection.mp4`（随仓库提交，5.5 MB）
- 来源：Intel IoT DevKit 官方示例视频 <https://github.com/intel-iot-devkit/sample-videos>（`people-detection.mp4`）
- 规格：768×432，12 fps，49.7 s，室内走廊固定机位俯视，行人进出画面

## 2. 类别与标注数量

只做一类目标：`person`。

| 类别 | train 标注框 | val 标注框 | 合计 |
| --- | --- | --- | --- |
| person | 360 | 36 | 396 |

- 标注方式：用 COCO 预训练的 YOLOv5s 自动初标（`conf≥0.35`、`imgsz 960`），再人工抽查约 30 帧修正漏检/误检；
  去掉短边 <12 px、面积占比 <0.06% 的框（避免出现明显偏小的框）。
- 框尺寸中位数：0.129×0.444（约 99×192 px），与画面中人物的实际大小吻合。

## 3. train / val 划分

- 抽帧：按原生 12 fps 抽满 596 张，其中 355 张画面里没有人，不参与训练，最终有目标 241 张。
- 划分：按**时间顺序**排好后，前 80% 作 train、后 20% 作 val；切点前后各丢弃 12 帧（1 s）作为间隔，
  因此验证集画面与训练集相邻但不重叠。
- 结果：train 193 张 / val 36 张。`scripts/04_check_dataset.py` 会用 md5 校验同一张图没有同时出现在两边。

## 4. 训练配置

| 项目 | 取值 |
| --- | --- |
| 代码 | YOLOv5 v7.0（`yolov5/` 目录，GPL-3.0） |
| 初始权重 | `yolov5s.pt`（COCO 预训练，微调） |
| 输入尺寸 / batch | 416 / 24 |
| epochs | 24（patience 10） |
| 优化器 | SGD，lr0 0.01，warmup 3 epoch，cos_lr 关闭 |
| 设备 | CPU（6 线程），单轮约 2.5 min，总用时约 1 h |

CPU 上 640 px 单轮要 8 min，24 轮之外还要大量时间，所以降到 416 px；单类、目标在画面中占比大，精度损失很小。

## 5. 验证集指标（提交的 best.pt）

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- |
| 0.997 | 1.000 | 0.995 | 0.782 |

验证集 36 张图、36 个目标。上表是用 `yolov5/val.py --weights best.pt --data data/hw.yaml --img 416 --workers 0`
在**提交的这份 best.pt**（14 MB，EMA 权重转 FP16 并去掉优化器状态）上复算得到的，训练曲线见 `results_curve.png`。

> 说明：验证集和训练集来自同一段视频的不同时间段，机位、光照、背景相同，所以指标偏高，
> 只代表“这段视频上的表现”，不代表换场景后的泛化能力。

## 6. 遇到的问题

1. **背景帧太多**：固定机位长时间空镜，直接抽帧会让数据集里大量是空图；按“有标注”过滤后再划分。
2. **自动标注的噪声**：伪标注偶尔给出过小的框（远景噪点）或同一人两个框；加了尺寸过滤并人工抽查修正。
3. **CPU 训练慢**：见上，降到 416 px / 24 epoch，并开启 `--cache ram` 减少读盘。
4. **新版本依赖兼容**：torch≥2.6 的 `weights_only` 默认值、numpy≥2 移除 `np.trapz`、Pillow≥10 移除
   `FreeTypeFont.getsize`，都会让 YOLOv5 v7.0 直接报错。仓库里对应位置做了兼容改动（都带注释，
   `torch>=2.6 compat` / `numpy>=2` / `Pillow>=10`）；另外 `requirements.txt` 把 setuptools 限制在 <81，
   否则 `pkg_resources` 不可用。

## 7. 复现步骤

```bash
pip install -r requirements.txt        # Python 3.12，CPU 亦可

# 1) 抽帧（12 fps，不做去重，保留全部时间点）
python scripts/01_extract_frames.py --video assets/people-detection.mp4 --out dataset/frames --fps 12 --dedupe 0
# 2) 自动初标（yolov5s.pt 缺失时会自动下载）
python scripts/02_label_frames.py --weights yolov5s.pt --images dataset/frames --out dataset/labels_raw --classes person
# 3) 按时间顺序划分 train/val
python scripts/03_split_dataset.py --frames dataset/frames --labels dataset/labels_raw --out dataset --train-ratio 0.8 --classes person
# 4) 数据集自检（nc/names、标注合法性、验证集是否混入训练集）
python scripts/04_check_dataset.py --data data/hw.yaml --dataset dataset
# 5) 训练（超参与提交的 best.pt 一致）
python scripts/05_train.py --data data/hw.yaml --weights yolov5s.pt
# 6) 推理出 result.mp4 与 3 帧截图
python scripts/06_infer_video.py --weights best.pt --video assets/people-detection.mp4 --out result.mp4 --start 14.5 --duration 24
```

数据集本身未提交：`dataset/` 已被 `.gitignore` 忽略，按上面 1~3 步可在本地重建出同样的 train/val。

## 8. 目录结构

```
best.pt              # 训练产出权重（14 MB，FP16 EMA）
result.mp4           # 推理视频（24 s，768×432）
frame_01/02/03.png   # 从 result.mp4 里截取的 3 帧（每段挑检测框最多的一帧）
results_curve.png    # 训练曲线（loss / mAP）
README.md
assets/              # 源视频
data/hw.yaml         # 数据集配置（nc=1, names=['person']）
scripts/01..06       # 抽帧 / 标注 / 划分 / 自检 / 训练 / 推理
yolov5/              # YOLOv5 v7.0 源码（含 LICENSE）
runs/                # 训练日志、曲线（未提交）
dataset/             # 本地数据集（未提交）
```
