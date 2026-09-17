# 基于 ResNet-18 的 Oxford-IIIT Pet 细粒度分类

本仓库用预训练的 ResNet-18 在 Oxford-IIIT Pet 数据集上做细粒度分类，提供一套基线配置和三个单变量消融配置（MixUp、冻结骨干、Label Smoothing），并包含混淆矩阵与 Grad-CAM 可视化。

## 数据集
Oxford-IIIT Pet 数据集包含 37 个猫狗品种、7349 张带标注图片。
首次训练会自动下载数据集到 `datasets/oxford-iiit-pet/`，不需要手动准备：

```
datasets/oxford-iiit-pet/
├── images/            7390 张 jpg，约 775 MB
├── annotations/       约 31 MB，包含 trainval.txt、test.txt、list.txt 与 trimaps/
├── images.tar.gz      下载中间产物，自动下载时解压后不会删除
└── annotations.tar.gz
```

两个压缩包合计约 800 MB，解压后仍留在目录里，需要回收空间时手动删除即可，不影响后续运行。

数据划分在第一次运行时生成并写入 `datasets/oxford_iiit_pet_split.json`：合并官方 trainval.txt 与 test.txt 的 7349 条记录，按 70/15/15 分层抽样。

## 一键复现

### 本地环境
Python 3.11 或更高版本，在仓库根目录执行：

```bash
pip install -r requirements.txt
```

Windows 下需要 GPU 时，还需要用下面命令单独装 cuda 版本 torch 和 torchvision：

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu126
```

训练基线并用测试集评估：

```bash
# 训练，权重存到 experiments/baseline/checkpoints/best_model.pth
python train.py --config config/baseline.yml

# 评估，生成混淆矩阵与两张 Grad-CAM 图
python evaluate.py \
  --checkpoint experiments/baseline/checkpoints/best_model.pth \
  --output experiments/baseline/evaluation

# 查看训练曲线
tensorboard --logdir experiments/baseline/logs
```

一次跑完全部四组实验：

```bash
for cfg in baseline mixup frozen label_smoothing; do
  python train.py --config config/$cfg.yml
  python evaluate.py --checkpoint experiments/$cfg/checkpoints/best_model.pth \
                     --output experiments/$cfg/evaluation
done
```

### Colab 或 Kaggle 上复现

克隆仓库并装依赖：

```python
!git clone https://github.com/<用户名>/pet-classifier.git
%cd pet-classifier
!pip install -r requirements.txt
```

全部四组实验并在测试集上评估：

```python
%%bash
for cfg in baseline mixup frozen label_smoothing; do
  python train.py --config config/$cfg.yml
  python evaluate.py --checkpoint experiments/$cfg/checkpoints/best_model.pth \
                     --output experiments/$cfg/evaluation
done
```
重跑同一实验前要先删除对应的 `experiments/<实验名>/` 目录。

## 实验设置与消融

固定超参：ResNet-18 预训练、AdamW（lr=1e-4、weight_decay=1e-4）、CrossEntropyLoss、batch size 32、15 epoch。训练集用 Resize(256) + RandomCrop(224) + RandomHorizontalFlip，验证集与测试集用 Resize(256) + CenterCrop(224)，两者都用 ImageNet 均值方差归一化。CUDA 上开启 AMP 混合精度。

| 配置文件 | 改动 | 对照目的 |
| :--- | :--- | :--- |
| `config/baseline.yml` | 基础增强（RandomCrop + 随机水平翻转） | 基线 |
| `config/mixup.yml` | `augmentation.mixup: true`（α=0.2） | 数据增强是否减轻过拟合 |
| `config/frozen.yml` | `model.freeze_backbone: true` | 冻结骨干与全参数微调的收敛速度和精度权衡 |
| `config/label_smoothing.yml` | `train.label_smoothing: 0.1` | 对相似品种的混淆是否有改善 |

## 实验结果

测试集 1103 张，`best_model.pth` 由验证集 Top-1 选出。

| 实验配置 | Top-1 Acc (%) | Top-5 Acc (%) | Macro-F1 | 训练轮数/耗时 |
| :--- | :--- | :--- | :--- | :--- |
| 1. Baseline（基础增强 + 全参数微调） | 90.66 | 99.46 | 0.9061 | 15 ep / 115.6s |
| 2. 改进组 A（+ MixUp α=0.2） | 91.12 | 99.18 | 0.9101 | 15 ep / 109.3s |
| 3. 改进组 B（冻结骨干，仅微调分类头） | 90.30 | 99.55 | 0.9022 | 15 ep / 80.2s |
| 4. 改进组 C（+ Label Smoothing 0.1） | 92.48 | 98.82 | 0.9245 | 15 ep / 108.1s |

验证集最佳 Top-1 与对应轮次：Baseline 92.56%（第 4 轮）、MixUp 92.56%（第 5 轮）、冻结骨干 91.38%（第 12 轮）、Label Smoothing 93.01%（第 15 轮）。

耗时在本机四组串行测得，只用于组间比较。冻结骨干省时间但收敛更慢，最佳轮次从第 4 轮推迟到第 12 轮；Label Smoothing 收益最大，Macro-F1 提升 0.0184。

基线上最容易混淆的三对品种各 7 例：Chihuahua 与 Miniature Pinscher、Bengal 与 Egyptian Mau、American Pit Bull Terrier 与 Staffordshire Bull Terrier。Label Smoothing 把最后一对降到 4 例，Bengal 与 Egyptian Mau 仍是 7 例。

## 目录结构

```
pet-classifier/
├── config/
│   ├── baseline.yml              基线配置
│   ├── mixup.yml                 MixUp 消融
│   ├── frozen.yml                冻结骨干消融
│   └── label_smoothing.yml       Label Smoothing 消融
├── data/
│   └── datasets.py               下载、分层划分、Transform、DataLoader
├── models/
│   └── model.py                  ResNet-18 骨干与分类头替换，支持冻结
├── utils/
│   ├── config.py                 配置系统
│   ├── metrics.py                Top-1/Top-5/Macro-F1、混淆矩阵、训练曲线
│   └── gradcam.py                Grad-CAM 热力图
├── train.py                      训练入口
├── evaluate.py                   评估与可视化入口
├── requirements.txt
└── README.md
```

## 输出产物

每次运行 `train.py` 新建 `experiments/<实验名>/`：

| 文件 | 内容 |
| :--- | :--- |
| `checkpoints/best_model.pth` | 验证集 Top-1 最高的权重，附带配置、类别表与划分索引 |
| `checkpoints/last_model.pth` | 最后一轮的权重 |
| `config.yaml` | 本次运行解析后的完整配置，不含相对 base 引用 |
| `history.csv` | 每轮的 train_loss、train_accuracy、val_top1、val_top5、val_macro_f1、val_loss、lr |
| `curves.png` | 由 history.csv 绘制 |
| `split.json` | 本次使用的数据划分索引 |
| `training.json` | 最佳验证 Top-1、实际轮数、耗时、设备与硬件型号 |
| `logs/` | TensorBoard 事件文件 |

`evaluate.py` 在 `--output` 目录写出：

| 文件 | 内容 |
| :--- | :--- |
| `metrics.json` | Top-1、Top-5、Macro-F1、loss、评估轮次与最易混淆的三对品种 |
| `predictions.csv` | 每张图的全局索引、真实标签、预测标签 |
| `confusion_matrix.png` | 37×37 混淆矩阵 |
| `gradcam_correct.png` | 预测正确的样本及其 Grad-CAM |
| `gradcam_incorrect.png` | 预测错误的样本及其 Grad-CAM |

## 命令行参数

两个脚本的参数形式如下，方括号内为可省略的参数：

```text
train.py --config 配置文件 [--override 配置项=值 ...]
```

`train.py` 的参数：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--config` | `config/baseline.yml` | 配置文件路径，可写绝对路径，也可写相对于 train.py 所在目录的路径 |
| `--override` | 无 | 覆盖配置里的任意项，可一次给多项，也可重复使用，优先级高于配置文件 |

```text
evaluate.py --checkpoint 权重路径 --output 输出目录 [--split val|test] [--device auto|cpu|cuda] [--data-root 数据集目录]
```
`evaluate.py` 的参数：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--checkpoint` | 必填 | 权重路径 |
| `--output` | 必填 | 输出目录，必须不存在 |
| `--split` | `test` | `val` 或 `test` |
| `--device` | `auto` | `auto` 时优先用 CUDA |
| `--data-root` | 配置文件里的值 | 覆盖数据集根目录 |

## 复现性

- `seed=42`，同时设置 random、numpy、torch 与 CUDA 的种子，DataLoader worker 也通过 `worker_init_fn` 播种
- `runtime.deterministic: true`，开启 cuDNN 确定性算法
- 数据划分写入 JSON 并复用，`data/datasets.py` 会校验三份划分互斥且正好覆盖全部 7349 张，也会校验标注文件顺序与 torchvision 内部顺序一致，不一致直接报错
- 训练过程只使用训练集与验证集，测试集在方案定稿后评估一次
- 完整训练重复执行两遍，四组实验的验证集与测试集指标逐位一致