# CelebA 公平性图像分类 —— Fair Contrastive Learning

## 1. 项目简介

本项目基于 CelebA 人脸属性数据集，研究图像分类任务中的**公平性（Fairness）**问题。具体任务是预测目标属性 `Blond_Hair`（是否金发），同时消除模型对敏感属性 `Male`（性别）的虚假依赖。

### 1.1 问题背景

CelebA 数据集中存在严重的虚假相关性（Spurious Correlation）：绝大多数金发样本是女性，金发男性极为稀少。标准的 ERM（经验风险最小化）训练方式会让模型"走捷径"，将"女性特征"作为"金发"的判据，导致对金发男性（Blond Male）的分类准确率极低。

训练集的样本分布清楚地反映了这一点：

| 群组 | 目标标签 | 敏感属性 | 样本数 | 占比 |
|---|---|---|---|---|
| NonBlond_Female | 非金发 | 女性 | 71,629 | 44.0% |
| NonBlond_Male | 非金发 | 男性 | 66,874 | 41.1% |
| Blond_Female | 金发 | 女性 | 22,880 | 14.1% |
| **Blond_Male** | **金发** | **男性** | **1,387** | **0.85%** |

Blond_Male 仅占全部训练样本的 0.85%，与最大群组的比例约为 1:50。

### 1.2 解决方案

采用 **Fair Supervised Contrastive Learning**（公平监督对比学习）方法：

- 构建双头模型（嵌入投影头 + 分类头）
- 在嵌入空间中，将"同一目标标签、不同敏感属性"的样本拉近（跨性别正对），迫使表征忽略性别信息
- 结合组平衡采样，确保少数群组在训练中获得充分的梯度信号

---

## 2. 项目结构

```
CelebA/
├── config.py              # 超参数与路径配置
├── dataset.py             # 数据集加载 + 组平衡采样器
├── model.py               # ResNet-18 双头模型
├── loss.py                # CE + FairSupCon 损失函数
├── train.py               # 训练主循环（baseline / debias 模式）
├── eval.py                # 评估：整体 + 分组准确率
├── requirements.txt       # 依赖
├── checkpoints/           # 模型权重保存目录
│   ├── best_baseline.pt
│   └── best_debias.pt
└── datasets/              # 数据目录
    ├── list_attr_celeba.csv
    ├── list_eval_partition.csv
    └── img_align_celeba/  # CelebA 对齐后的人脸图片
```

---

## 3. 环境配置

### 3.1 依赖安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
torch>=2.0
torchvision>=0.15
pandas
Pillow
scikit-learn
```

### 3.2 数据准备

从 [CelebA 官方页面](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 下载以下文件，放入 `datasets/` 目录：

1. `img_align_celeba/` — 对齐裁剪后的人脸图片（202,599 张）
2. `list_attr_celeba.csv` — 40 个属性标签（原始值为 -1/1，代码中转为 0/1）
3. `list_eval_partition.csv` — 官方 train/val/test 划分（0/1/2）

---

## 4. 方法详解

### 4.1 模型架构（`model.py`）

基于预训练 ResNet-18，去掉原始全连接层后接两个头：

```
输入图片 (224x224x3)
    │
    ▼
ResNet-18 backbone → 512 维特征向量 (feat)
    │                           │
    ▼                           ▼
Projection Head             Classification Head
(512→512→ReLU→128)         (512→2)
    │                           │
    ▼                           ▼
L2 归一化 embedding          logits (二分类)
(用于对比损失)               (用于交叉熵损失)
```

- **Projection Head**：两层 MLP，将 512 维特征映射到 128 维嵌入空间，经 L2 归一化后用于计算对比损失
- **Classification Head**：单层线性层，输出 2 类 logits

### 4.2 损失函数（`loss.py`）

总损失公式：

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{FairSupCon}$$

#### 交叉熵损失 (CE)

标准的二分类交叉熵损失，用于监督分类任务。

#### 公平监督对比损失 (FairSupCon)

核心思想是在嵌入空间中消除 sensitive attribute 的影响：

- **正对（Positive Pairs）**：同一目标标签 + **不同**敏感属性的样本对（例如：金发女性 ↔ 金发男性）
- **负对（Negative Pairs）**：不同目标标签的所有样本对

通过最大化正对的相似度、最小化负对的相似度，模型被迫学习到"只看发色、不看性别"的表征。

具体计算流程：

1. 计算所有样本对的余弦相似度矩阵，除以温度参数 τ
2. 构建正对掩码：`pos_mask = same_target × diff_sensitive`
3. 对数值做稳定化处理（减去行最大值）
4. 计算 InfoNCE 形式的对比损失

当 `sensitives=None`（baseline 模式）时，退回标准 SupCon：正对 = 同一 target 的所有样本对。

### 4.3 数据加载与采样策略（`dataset.py`）

#### 数据预处理

- 训练集：RandomResizedCrop(224) + RandomHorizontalFlip + ImageNet 归一化
- 验证/测试集：Resize(256) + CenterCrop(224) + ImageNet 归一化

#### 组平衡采样 (Group-Balanced Sampling)

针对极端的组别不平衡，实现了基于 `WeightedRandomSampler` 的组平衡采样：

每个样本的采样权重 = 1 / 该组样本数

这使得在每个 epoch 中，4 个群组被大致均匀地采样。Blond_Male 的采样概率被提升约 50 倍，确保模型在训练中能充分学习到少数群组的特征。

仅在 debias 模式下启用，baseline 模式使用标准随机采样。

### 4.4 训练策略（`train.py`）

#### 差分学习率

- ResNet-18 backbone：`lr = 1e-5`（较小，保留预训练特征）
- Projection Head + Classification Head：`lr = 1e-4`（较大，快速适应新任务）

#### 学习率调度

使用 Cosine Annealing 学习率衰减，T_max = 总 epoch 数。

#### Early Stopping

以验证集上的 Worst-Group Accuracy (WGA) 为监控指标，连续 7 个 epoch 无提升则停止训练。每次 WGA 创新高时保存模型。

### 4.5 评估指标（`eval.py`）

- **Overall Accuracy**：全部样本的分类准确率
- **Group Accuracy**：4 个群组各自的分类准确率
- **Worst-Group Accuracy (WGA)**：4 个群组中最低的准确率，是衡量公平性的核心指标

---

## 5. 使用方法

### 5.1 训练 Baseline 模型

```bash
python train.py --mode baseline
```

Baseline 模式下：λ=0（无对比损失），标准随机采样，等同于普通 ERM 训练。

### 5.2 训练 Debias 模型

```bash
python train.py --mode debias
```

Debias 模式下：λ=0.2，启用组平衡采样，启用跨 sensitive 正对的 FairSupCon 损失。

#### 可选参数

```bash
python train.py --mode debias \
    --lambda_con 0.3 \   # 对比损失权重（默认 0.2）
    --epochs 50 \        # 最大 epoch 数（默认 50）
    --lr 1e-4 \          # head 学习率（默认 1e-4）
    --bs 128 \           # batch size（默认 128）
    --patience 7         # early stopping patience（默认 7）
```

### 5.3 评估模型

```bash
# 评估 debias 模型在测试集上的表现
python eval.py --checkpoint checkpoints/best_debias.pt --split test

# 评估 baseline 模型在验证集上的表现
python eval.py --checkpoint checkpoints/best_baseline.pt --split val
```

---

## 6. 实验过程与迭代记录

### 6.1 V1：初始版本 — 基础框架搭建

**做了什么：** 搭建了完整的模块化代码框架。

- `config.py`：集中管理路径和超参数
- `dataset.py`：CelebA 数据集读取，返回 (image, target, sensitive, group) 四元组
- `model.py`：ResNet-18 双头模型（嵌入 + 分类）
- `loss.py`：CrossEntropy + 标准 SupCon（正对仅按 target label 构建）
- `train.py`：训练循环，支持 baseline/debias 两种模式切换
- `eval.py`：分组准确率评估

**初始超参数：**

| 参数 | 值 |
|---|---|
| epochs | 10 |
| lr | 1e-4（全局统一） |
| batch_size | 128 |
| λ (LAMBDA_CON) | 1.0 |
| 采样策略 | 标准随机采样 |
| early stopping | 无 |

### 6.2 V1 实验结果

#### Baseline（λ=0，标准 ERM）

```
训练 10 个 epoch，best WGA 出现在 epoch 9

[9/10]  loss=0.0519  acc=95.40%  wga=41.76%
  NonBlond_Female: 95.93%
  NonBlond_Male:   99.49%
  Blond_Female:    85.46%
  Blond_Male:      41.76%   ← 最差组

Best WGA = 41.76%
```

#### Debias（λ=1.0，标准 SupCon）

```
训练 10 个 epoch，best WGA 出现在 epoch 2

[2/10]  loss=4.7302  acc=95.08%  wga=48.35%
  NonBlond_Female: 93.10%
  NonBlond_Male:   99.24%
  Blond_Female:    91.93%
  Blond_Male:      48.35%   ← 最差组

Best WGA = 48.35%（之后逐轮下降到 35%，极不稳定）
```

#### V1 问题分析

1. **组别不平衡未处理**：Blond_Male 仅 1,387 样本（占 0.85%），标准采样下模型几乎忽略此组
2. **SupCon 未利用 sensitive attribute**：正对仅按 target 构建（same target = 正对），没有显式消除 gender 信息，只是做了普通的有监督对比学习
3. **λ=1.0 导致 loss 尺度失衡**：CE loss ≈ 0.1，SupCon loss ≈ 4.6，对比损失远大于分类损失，分类头的梯度信号被淹没
4. **无 early stopping**：最优模型在 epoch 2 就出现，后续 8 个 epoch 浪费训练时间且 WGA 持续退化

### 6.3 V2：全面改进

针对 V1 的问题，做了以下改动：

#### 改动 1：组平衡采样（`dataset.py`）

新增 `_group_balanced_sampler()` 函数，使用 `WeightedRandomSampler`，每个样本的采样权重 = 1/该组样本数。Blond_Male 的采样概率从 0.85% 提升到 25%（约 50 倍）。

`get_loader()` 新增 `balanced` 参数，debias 模式自动启用。

#### 改动 2：降低对比损失权重（`config.py`）

`LAMBDA_CON`：1.0 → **0.2**，使 CE 和 SupCon 的梯度量级接近。

#### 改动 3：真正的 Fair SupCon（`loss.py`）

`FairSupConLoss.forward()` 新增 `sensitives` 参数：

- 传入 sensitives 时：**正对 = same target & different sensitive**（跨性别对比，强制表征忽略 gender 信息）
- 不传入时：退回标准 SupCon（same target = 正对），baseline 不受影响

`TotalLoss.forward()` 同步透传 `sensitives`。

#### 改动 4：差分学习率（`config.py` + `train.py`）

- 新增 `LR_BACKBONE = 1e-5`（比 head 的 1e-4 慢 10 倍）
- 使用 `torch.optim.Adam` 的参数组功能分别设置学习率
- 新增 `CosineAnnealingLR` 学习率调度器

#### 改动 5：Early Stopping（`config.py` + `train.py`）

- 新增 `PATIENCE = 7`
- `NUM_EPOCHS`：10 → **50**（提供更大的搜索空间，配合 early stopping）
- 以验证集 WGA 为监控指标，连续 7 个 epoch 无提升即停止

### 6.4 V2 实验结果

#### Baseline（改进后）

```
训练至 epoch 9 触发 early stopping

[2/50]  loss=0.1071  acc=95.29%  wga=43.96%
  NonBlond_Female: 94.20%
  NonBlond_Male:   99.38%
  Blond_Female:    89.98%
  Blond_Male:      43.96%   ← 最差组

Best WGA = 43.96%（相比 V1 的 41.76% 略有提升）
```

#### Debias（改进后）

```
训练至 epoch 8 触发 early stopping

[1/50]  loss=1.1014  acc=91.95%  wga=87.91%
  NonBlond_Female: 88.45%
  NonBlond_Male:   94.38%
  Blond_Female:    95.58%
  Blond_Male:      87.91%   ← 最差组

Best WGA = 87.91%（相比 V1 的 48.35% 提升了 39.56 个百分点）
```

### 6.5 结果对比总结

| 指标 | V1 Baseline | V1 Debias | V2 Baseline | V2 Debias |
|---|---|---|---|---|
| **Best WGA** | 41.76% | 48.35% | 43.96% | **87.91%** |
| **Overall Acc** | 95.40% | 95.08% | 95.29% | 91.95% |
| Blond_Male | 41.76% | 48.35% | 43.96% | **87.91%** |
| Blond_Female | 85.46% | 91.93% | 89.98% | 95.58% |
| NonBlond_Male | 99.49% | 99.24% | 99.38% | 94.38% |
| NonBlond_Female | 95.93% | 93.10% | 94.20% | 88.45% |
| 组间最大差距 | 57.73pp | 50.89pp | 55.42pp | **7.13pp** |

**关键结论：**

1. V2 Debias 的 WGA 从 48.35% 跃升至 **87.91%**，提升幅度接近 40 个百分点
2. 四组准确率差距从 ~57pp 缩小到仅 **~7pp**，模型公平性大幅提升
3. Overall Accuracy 仅下降约 3pp（95% → 92%），这是合理的 fairness-accuracy tradeoff
4. 三项核心改动（组平衡采样 + 跨 sensitive 正对 + 差分学习率）共同发挥了作用

---

## 7. 核心超参数一览

| 参数 | 值 | 说明 |
|---|---|---|
| `BATCH_SIZE` | 128 | 批次大小 |
| `NUM_EPOCHS` | 50 | 最大训练轮数 |
| `LR` | 1e-4 | Head 学习率 |
| `LR_BACKBONE` | 1e-5 | Backbone 学习率 |
| `WD` | 1e-4 | 权重衰减 |
| `EMBED_DIM` | 128 | 嵌入空间维度 |
| `TEMPERATURE` | 0.07 | 对比损失温度参数 |
| `LAMBDA_CON` | 0.2 | 对比损失权重 |
| `PATIENCE` | 7 | Early stopping 容忍轮数 |
| `SEED` | 42 | 随机种子 |