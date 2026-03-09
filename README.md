# Fair Classification on CelebA — Q1 Action Plan

## 1. De-biasing Method

**Task:** Predict `Blond_Hair` on CelebA. **Blond_Male** is only **0.85 %** of training data, so ERM learns a shortcut: *"blond ≈ female"*.

**Method:** FairSupCon + group-balanced resampling, with ablation to see what each part does. Group DRO (Sagawa et al., ICLR 2020) as other SOTA.


| Stage | **Method**              | **Sampling**   | **Loss**   | **What it fixes?**                 |
| ----- | ----------------------- | -------------- | ---------- | ---------------------------------- |
| ①     | ERM                     | Uniform        | CE         | Baseline，expose shortcut           |
| ②     | + Resampling            | Group-balanced | CE         | By based on data level             |
| ③     | + FairSupCon            | Uniform        | CE + λ·FSC | By based on representational level |
| ④     | Resampling + FairSupCon | Group-balanced | CE + λ·FSC | Combine together                   |
| ⑤     | vs Group DRO            | Uniform        | DRO        | Compared to other SOTA             |


This is a **2×2 ablation Matrix**：


|                  | **不 Resampling**   | **Resampling**     |
| ---------------- | ------------------ | ------------------ |
| **不 FairSupCon** | ① ERM              | ② ERM + Resampling |
| **FairSupCon**   | ③ ERM + FairSupCon | ④ **Final Method** |


This design is called **factorial ablation**.

**Expected**: 

*① ERM ≪ ③ FairSupCon only < ② Resampling only < ④ Combined ≈ or > ⑤ Group DRO*

## 2. Formulation

### 2.1 Background — SupCon

SupCon (Khosla et al., NeurIPS 2020) uses labels to pick positive pairs. For anchor $i$, any sample with the same label is a positive:

$$\mathcal{P}_{\text{SupCon}}(i) = \left\lbrace i \neq j \mid y_i = y_j \right\rbrace$$

$$\mathcal{L}*{\text{SupCon}} = -\frac{1}{|\mathcal{B}|}\sum*{i \in \mathcal{B}} \frac{1}{|\mathcal{P}*{\text{SupCon}}(i)|} \sum*{j \in \mathcal{P}*{\text{SupCon}}(i)} \log \frac{\exp(\text{sim}(i,j) / \tau)}{\sum*{k \neq i} \exp(\text{sim}(i,k) / \tau)}$$

where $\text{sim}(i,j) = \mathbf{z}_i \cdot \mathbf{z}_j$ (cosine similarity, L2-normalized). Same-class pulled closer, different-class pushed apart.

**Problem:** SupCon treats all same-label pairs equally. Most Blond samples are female, so it clusters by *gender* not *hair color*.

### 2.2 FairSupCon Loss

We only pair samples with **same label but different sensitive attribute**:

$$\mathcal{P}_{\text{Fair}}(i) = \left\lbrace i \neq j \mid y_i = y_j \wedge s_i \neq s_j \right\rbrace$$

e.g. Blond_Female ( $y=1, s=0$ ) only pairs with BlondMale ( $y=1, s=1$ ), never another BlondFemale. This forces the encoder to learn hair-color features instead of gender.

But if we keep the standard denominator, same-class same-attribute samples get pushed apart, which breaks clustering. So we fix the denominator too.

Negative set — everything with a different label:

$$\mathcal{N}(i) = \left\lbrace k \mid y_k \neq y_i \right\rbrace$$

Denominator set — union of cross-attribute positives and negatives:

$$\mathcal{D}(i) = \mathcal{P}_{\text{Fair}}(i) \cup \mathcal{N}(i)$$

Same-label same-attribute samples ( $y_k = y_i \wedge s_k = s_i$ ) are left out of $\mathcal{D}(i)$ — not pulled, not pushed, they just cluster on their own.

$$\mathcal{L}*{\text{FSC}} = - \frac{1}{|\mathcal{B}|} \sum*{i \in \mathcal{B}} \frac{1}{|\mathcal{P}*{\text{Fair}}(i)|} \sum*{j \in \mathcal{P}_{\text{Fair}}(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}*j / \tau)}{\sum*{k \in \mathcal{D}(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}$$

### 2.3 Total Loss & Hyperparameters

$$\mathcal{L}*{\text{total}} = \mathcal{L}*{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{FSC}}$$


| Symbol    | What it is                 | Value                |
| --------- | -------------------------- | -------------------- |
| $\lambda$ | FairSupCon weight          | 0.2 (0 for baseline) |
| $\tau$    | Temperature                | 0.07                 |
| $d$       | Projection head output dim | 128                  |
| $B$       | Batch size                 | 128                  |


## 3. Experiment Design


| Comparison                        | Why                                         |
| --------------------------------- | ------------------------------------------- |
| ERM vs. ERM + Resampling          | See how much balancing alone helps          |
| ERM + Resampling vs. + FairSupCon | See what contrastive de-biasing adds on top |
| FairSupCon vs. Group DRO          | Compare with a well-known de-biasing method |


**Main metric:** Worst-Group Accuracy (WGA) = $\min \text{Acc}_g$.

## 4. Responsibilities


| Member      | What they're doing                                           |
| ----------- | ------------------------------------------------------------ |
| **Vaibhav** | Baseline ERM; Group DRO baseline; resampling                 |
| **Huayi**   | FairSupCon loss design & implementation; fairness evaluation |
| **Matthew** | FairSupCon integration & tuning; ablation experiments        |


## 5. Timeline & Milestones


| Date            | Milestone                                               |
| --------------- | ------------------------------------------------------- |
| **Mar 6 (Q1)**  | Problem defined; action plan finalized                  |
| **Mar 13 (Q2)** | Pipeline working; baseline + FairSupCon initial results |
| **Mar 20**      | Hyperparameter sweep; ablation analysis                 |
| **Mar 27 (Q3)** | Full comparison; fairness evaluation; final report      |


## 6. 训练集 / 测试集准确率高且公平性高的做法

目标：训练准确率高、测试准确率高、公平性（WGA / EOD）也好。

- **用对 checkpoint**  
训练会保存多份 best：`best_<tag>_wga.pt`（最差组最好）、`best_<tag>_eod.pt`（机会公平最好）、`best_<tag>_tradeoff.pt`（WGA 与 EOD 折中）。做测试集评估时用这些，不要用最后一轮的权重；汇报时优先用 `best_*_wga.pt` 或 `best_*_tradeoff.pt`。
- **超参建议**  
  - `lambda_con` 在 1.0～2.0 之间通常能兼顾准确率与公平性；过大可能压准确率，过小公平性易被 CE 盖掉。  
  - 总 epoch 按需设置；若后期出现「整体准确率↑、WGA↓、EOD↑」，可提前停训并选用已保存的 best checkpoint。

**使用方式：**

- 只打一行汇总（与之前一致）：
```
python eval.py --checkpoint <path>
```
- 同时看详细公平性报告：
```
python eval.py --checkpoint <path> --report
```
