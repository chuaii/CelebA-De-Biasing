# CelebA 去偏分类 —— Learning from Failure (LfF)

## 1. 方法简介

本模块实现了 **Learning from Failure (LfF)** 去偏方法（Nam et al., 2020），作为对主项目 Fair Contrastive Learning 方法的**对比实验**。

### 1.1 问题回顾

CelebA 数据集中存在严重的虚假相关性：绝大多数金发样本是女性，金发男性（Blond Male）仅占训练集的 0.85%。标准 ERM 训练会让模型将"女性特征"作为"金发"的判据，导致 Blond Male 组准确率极低。

| 群组 | 样本数 | 占比 | 偏见属性 |
|---|---|---|---|
| NonBlond_Female | 71,629 | 44.0% | 偏见对齐 |
| NonBlond_Male | 66,874 | 41.1% | 偏见对齐 |
| Blond_Female | 22,880 | 14.1% | 偏见对齐 |
| **Blond_Male** | **1,387** | **0.85%** | **偏见冲突** |

### 1.2 核心思想

LfF 的核心假设是：**一个故意被偏见训练的模型，其失败的样本恰好就是偏见冲突样本**。

直觉解释：
- 偏见对齐样本（如 Blond Female）对有偏模型来说很"简单" → 低损失
- 偏见冲突样本（如 Blond Male）对有偏模型来说很"困难" → 高损失
- 利用这个损失差异，可以自动识别出偏见冲突样本，无需事先知道偏见属性的标签

---

## 2. 算法流程

LfF 采用三阶段管线：

### Stage 1: 训练有偏模型 (Biased Model)

使用 **Generalized Cross Entropy (GCE)** 损失训练 ResNet-18：

$$\mathcal{L}_{GCE}(p, y) = \frac{1 - p_y^q}{q}$$

其中 $p_y$ 为模型对真实类别的预测概率，$q \in (0, 1]$ 为截断参数。

**GCE 的关键性质**：相比标准 CE，GCE 会"截断"高损失样本的梯度贡献。这意味着：
- 简单样本（偏见对齐）主导优化方向
- 困难样本（偏见冲突）的梯度被抑制
- 结果：模型快速学到数据中的偏见捷径

| $p_y$ (预测置信度) | CE Loss | GCE Loss (q=0.7) |
|---|---|---|
| 0.99 (高置信) | 0.01 | 0.01 |
| 0.50 (不确定) | 0.69 | 0.55 |
| 0.01 (预测错误) | 4.61 | 1.36 |

可以看到，GCE 显著压缩了高损失区间的范围（4.61 → 1.36），使困难样本的梯度信号大幅减弱。

配合较少的训练轮数（默认 5 epoch）和较大的学习率（1e-3），确保模型只学到最"简单"的偏见模式。

### Stage 2: 失败识别与权重计算

1. 将整个训练集通过有偏模型前向推理（使用 eval 变换，无随机增强）
2. 计算每个样本的标准 CE 损失 $\ell_i$
3. 归一化得到样本权重：

$$w_i = \text{clamp}\left(\frac{\ell_i}{\bar{\ell}},\ w_{min},\ w_{max}\right)$$

其中 $\bar{\ell}$ 为全局平均损失。

**预期结果**：
- Blond_Male 组（偏见冲突）：平均权重 >> 1
- 其他组（偏见对齐）：平均权重 ≈ 1 或 < 1

### Stage 3: 训练去偏主模型 (Debiased Model)

初始化全新的 ResNet-18，使用**加权交叉熵损失**训练：

$$\mathcal{L}_{weighted} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot \text{CE}(f(x_i), y_i)$$

高权重的偏见冲突样本在损失中获得更大贡献，迫使主模型：
- 不能仅依赖性别特征做判断（因为 Blond Male 的权重被显著提升）
- 必须学习真正基于发色的特征表示

配合差分学习率（backbone 1e-5, head 1e-4）、Cosine Annealing 调度器和 Early Stopping（基于验证集 WGA）。

---

## 3. 文件结构

```
CelebA/
├── LfF_config.py      # LfF 超参数配置（复用主项目路径定义）
├── LfF_loss.py        # GCE 损失 + 加权 CE 损失
├── LfF_model.py       # ResNet-18 单头分类器（有偏 & 主模型共用架构）
├── LfF_dataset.py     # IndexedCelebA 数据集包装器 + DataLoader 工厂
├── LfF_train.py       # 三阶段训练主流程
├── LfF_eval.py        # 分组评估（Overall / WGA / Unbiased Acc）
├── LfF_README.md      # 本文档
│
├── config.py          # (原有) 主项目配置
├── dataset.py         # (原有) CelebAFairness 数据集
├── model.py           # (原有) 双头 FairClassifier
├── loss.py            # (原有) CE + FairSupCon 损失
├── train.py           # (原有) Fair Contrastive Learning 训练
├── eval.py            # (原有) 主项目评估
│
└── checkpoints/
    ├── LfF_biased.pt      # Stage 1 有偏模型权重
    ├── best_LfF_main.pt   # Stage 3 最优去偏模型权重
    ├── best_baseline.pt   # (原有)
    └── best_debias.pt     # (原有)
```

### 模块依赖关系

```
LfF_config.py  ←── config.py（复用路径和群组定义）
     ↑
LfF_loss.py         （独立，仅依赖 PyTorch）
     ↑
LfF_model.py        （独立，仅依赖 torchvision）
     ↑
LfF_dataset.py ←── dataset.py（复用 CelebAFairness 和 transforms）
     ↑
LfF_eval.py    ←── LfF_config, LfF_dataset, LfF_model
     ↑
LfF_train.py   ←── 以上全部模块
```

---

## 4. 使用方法

### 4.1 训练 LfF 去偏模型（三阶段自动完成）

```bash
python LfF_train.py
```

全流程自动运行：训练有偏模型 → 计算样本权重 → 训练去偏主模型 → 测试集评估。

#### 可选参数

```bash
python LfF_train.py \
    --biased_epochs 5 \    # Stage 1 训练轮数（默认 5）
    --gce_q 0.7 \          # GCE 截断参数（默认 0.7）
    --biased_lr 1e-3 \     # Stage 1 学习率（默认 1e-3）
    --main_epochs 20 \     # Stage 3 最大训练轮数（默认 20）
    --main_lr 1e-4 \       # Stage 3 head 学习率（默认 1e-4）
    --bs 128 \             # batch size（默认 128）
    --patience 7           # early stopping patience（默认 7）
```

### 4.2 单独评估已训练模型

```bash
# 评估 LfF 去偏模型
python LfF_eval.py --checkpoint checkpoints/best_LfF_main.pt --split test

# 评估有偏模型（验证偏见确实存在）
python LfF_eval.py --checkpoint checkpoints/LfF_biased.pt --split test
```

### 4.3 预期输出格式

训练过程会输出：

```
============================================================
  Stage 1: Training Biased Model (GCE Loss)
============================================================
  [Biased] Epoch 1/5  GCE_loss=0.4823
  ...

==================================================
  Biased Model (Val)
==================================================
  Overall Accuracy : 95.xx%
  Worst-Group Acc  : 3x.xx%  (Blond_Male)     ← 有偏模型在 Blond_Male 上表现极差
  Unbiased Accuracy: 7x.xx%

============================================================
  Stage 2: Failure Identification & Re-weighting
============================================================
  [Weight Distribution by Group]
    NonBlond_Female     :  mean=0.8xx  ...
    NonBlond_Male       :  mean=0.7xx  ...
    Blond_Female        :  mean=1.1xx  ...
    Blond_Male          :  mean=3.xxx  ...    ← 偏见冲突组获得最高权重

============================================================
  Stage 3: Training Main Model (Weighted CE Loss)
============================================================
  [Main] Epoch 1/20  ...  wga=xx.xx%
  ...

==================================================
  LfF Debiased Model (Test)
==================================================
  Overall Accuracy : 9x.xx%
  Worst-Group Acc  : 6x-8x%    ← 相比有偏模型大幅提升
  Unbiased Accuracy: 8x.xx%
```

---

## 5. 超参数说明

| 参数 | 默认值 | 说明 | 调优建议 |
|---|---|---|---|
| `BIASED_EPOCHS` | 5 | 有偏模型训练轮数 | 越少偏见越强，但太少可能导致模型未收敛 |
| `BIASED_LR` | 1e-3 | 有偏模型学习率 | 较高以快速捕获偏见 |
| `GCE_Q` | 0.7 | GCE 截断参数 | 越大越聚焦简单样本；0.5~0.9 均可尝试 |
| `WEIGHT_CLAMP_MIN` | 0.1 | 最小样本权重 | 防止完全忽略简单样本 |
| `WEIGHT_CLAMP_MAX` | 10.0 | 最大样本权重 | 防止权重爆炸导致训练不稳定 |
| `MAIN_EPOCHS` | 20 | 主模型最大训练轮数 | 配合 early stopping |
| `MAIN_LR` | 1e-4 | 主模型 head 学习率 | — |
| `MAIN_LR_BACKBONE` | 1e-5 | 主模型 backbone 学习率 | 比 head 低 10x 保留预训练特征 |
| `MAIN_WD` | 1e-4 | 权重衰减 | — |
| `MAIN_PATIENCE` | 7 | Early stopping 容忍轮数 | — |

---

## 6. 与 Fair Contrastive Learning 的方法对比

| 维度 | Fair Contrastive Learning | Learning from Failure (LfF) |
|---|---|---|
| **核心机制** | 嵌入空间中拉近跨组正对 | 利用有偏模型失败自动识别偏见冲突样本 |
| **是否需要偏见标签** | 是（需要 sensitive attribute） | 否（自动从有偏模型推断） |
| **模型架构** | 双头（分类 + 投影） | 单头（仅分类） |
| **损失函数** | CE + FairSupCon | GCE（Stage 1）+ Weighted CE（Stage 3） |
| **采样策略** | 组平衡采样 | 标准采样 + 损失加权 |
| **训练流程** | 单阶段端到端 | 三阶段管线 |
| **适用场景** | 已知偏见属性时效果最优 | 偏见属性未知或未标注时仍可使用 |

### 关键区别

1. **Fair Contrastive Learning** 在嵌入空间层面操作，通过对比学习直接消除 sensitive 信息，需要事先知道哪个属性是偏见来源。

2. **LfF** 在样本层面操作，通过"先放大偏见、再反向利用"的策略自动发现偏见冲突样本。理论上即使不知道偏见属性是什么，也能起效（因为 GCE 天然会导致模型依赖最容易的特征，而最容易的特征往往就是偏见特征）。

---

## 7. 参考文献

- Nam, J., Cha, H., Ahn, S., Lee, J., & Shin, J. (2020). *Learning from Failure: De-biasing Classifier from Biased Classifier*. NeurIPS 2020.
