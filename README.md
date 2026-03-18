# Fair Classification on CelebA — Q1 Action Plan

## 1. De-biasing Method

**Task:** Study bias between attribute pairs in CelebA. ERM learns shortcuts when a target attribute co-occurs strongly with a sensitive attribute (e.g. most *Blond* samples are female, so the model learns *"blond ≈ female"* instead of hair color).

We select high-bias pairs from a full pairwise bias analysis of all 40 CelebA attributes:

| Target               | Sensitive        |  DPD  |  WMR  |
| -------------------- | ---------------- | :---: | :---: |
| Wearing_Lipstick     | Male             | 0.799 | 0.006 |
| Young                | Gray_Hair        | 0.761 | 0.002 |
| Smiling              | High_Cheekbones  | 0.686 | 0.145 |
| Blond_Hair           | Male             | —     | 0.009 |
| **Mouth_Slightly_Open** | **Smiling**   | **0.537** | **0.287** |

> DPD = \|P(T=1\|S=1) − P(T=1\|S=0)\|, WMR = min(group) / max(group). Current experiment: **Mouth\_Slightly\_Open × Smiling**.

![Bias Heatmap (DPD + min group)](bias_analysis/bias_heatmap.png)

**Experiment Design:** FairSupCon + group-balanced sampling, with ablation to see what each part does. Group DRO (Sagawa et al., ICLR 2020) as other SOTA.


| Stage | **Method**                                                                           | **Sampling**   | **Loss**       | **What it fixes?**                 |
| ----- | ------------------------------------------------------------------------------------ | -------------- | -------------- | ---------------------------------- |
| ①     | ERM                                                                                  | Unbalanced     | CE             | Baseline, expose shortcut          |
| ②     | + 4 group-balanced only: ( Oversampling / Undersampling / Reweighting / Focal Loss ) | Group-balanced | CE             | Data-level de-bias                 |
| ③     | + FairSupCon only                                                                    | Unbalanced     | CE + λ·FSC     | Representation-level de-bias       |
| ④     | Balanced (Oversampling / Reweighting) + FairSupCon                                   | Group-balanced | CE + λ·FSC     | Combine both levels                |
| ⑤     | vs Group DRO                                                                         | Unbalanced     | DRO            | Compared to other SOTA             |


This is a **2×2 ablation matrix**：


|                   | Unbalanced (Random) | Group-balanced           |
| ----------------- | ------------------- | ------------------------ |
| **No FairSupCon** | ① ERM               | ② ERM + Balanced Methods |
| **FairSupCon**    | ③ ERM + FairSupCon  | ④ **Final Method**       |


This design is called **factorial ablation**.

**Expected**:

*① ERM < ② Balanced only ≈ ③ FairSupCon only < ④ Combined ≈ ⑤ Group DRO*

> ② and ③ are roughly comparable; which is better depends on the bias type. Strong group imbalance (low WMR) favors ②; representation-level shortcut learning favors ③. ④ combining both is expected to be best among our methods.

## 2. Formulation

### 2.1 Background — SupCon

SupCon (Khosla et al., NeurIPS 2020) uses labels to pick positive pairs. For anchor $i$, any sample with the same label is a positive:

$$\mathcal{P}_{\text{SupCon}}(i) = \left\lbrace i \neq j \mid y_i = y_j \right\rbrace$$

$$\mathcal{L}_{\text{SupCon}} = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \frac{1}{|\mathcal{P}_{\text{SupCon}}(i)|} \sum_{j \in \mathcal{P}_{\text{SupCon}}(i)} \log \frac{\exp(\text{sim}(i,j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(i,k) / \tau)}$$

where $\text{sim}(i,j) = \mathbf{z}_i \cdot \mathbf{z}_j$ (cosine similarity, L2-normalized). Same-class pulled closer, different-class pushed apart.

**Problem:** SupCon treats all same-label pairs equally. If most *Mouth\_Slightly\_Open* samples are also *Smiling*, it clusters by smiling expression, not mouth position.

### 2.2 FairSupCon Loss

We only pair samples with **same label but different sensitive attribute**:

$$\mathcal{P}_{\text{Fair}}(i) = \left\lbrace i \neq j \mid y_i = y_j \wedge s_i \neq s_j \right\rbrace$$

e.g. MouthOpen\_NonSmiling ( $y=1, s=0$ ) only pairs with MouthOpen\_Smiling ( $y=1, s=1$ ), never another MouthOpen\_NonSmiling. This forces the encoder to learn mouth-position features instead of smiling expression.

But if we keep the standard denominator, same-class same-attribute samples get pushed apart, which breaks clustering. So we fix the denominator too.

Negative set — everything with a different label:

$$\mathcal{N}(i) = \left\lbrace k \mid y_k \neq y_i \right\rbrace$$

Denominator set — union of cross-attribute positives and negatives:

$$\mathcal{D}(i) = \mathcal{P}_{\text{Fair}}(i) \cup \mathcal{N}(i)$$

Same-label same-attribute samples ( $y_k = y_i \wedge s_k = s_i$ ) are left out of $\mathcal{D}(i)$ — not pulled, not pushed, they just cluster on their own.

$$\mathcal{L}_{\text{FSC}} = - \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{1}{|\mathcal{P}_{\text{Fair}}(i)|} \sum_{j \in \mathcal{P}_{\text{Fair}}(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \in \mathcal{D}(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}$$

### 2.3 Total Loss & Hyperparameters

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{FSC}}$$


| Symbol    | What it is                 | Value                |
| --------- | -------------------------- | -------------------- |
| $\lambda$ | FairSupCon weight          | 1.5 (0 for baseline) |
| $\tau$    | Temperature                | 0.07                 |
| $d$       | Projection head output dim | 128                  |
| $B$       | Batch size                 | 128                  |


## 3. Responsibilities


| Member      | What they're doing                                 |
| ----------- | -------------------------------------------------- |
| **Vaibhav** | Baseline ERM; Group DRO baseline                   |
| **Huayi**   | FairSupCon loss design & implementation            |
| **Matthew** | Group-balanced methods                             |
| **??**      | Fairness evaluation (Visualize results and report) |


## 4. Timeline & Milestones


| Date            | Milestone                                               |
| --------------- | ------------------------------------------------------- |
| **Mar 6 (Q1)**  | Problem defined; action plan finalized                  |
| **Mar 13 (Q2)** | Pipeline working; baseline + FairSupCon initial results |
| **Mar 20**      | Hyperparameter sweep; ablation analysis                 |
| **Mar 27 (Q3)** | Full comparison; fairness evaluation; final report      |


## Blond Hair vs Male — Pilot Experiment Results

### BlondHair × Male

![BlondHair vs Male comparison](results/blond_hair_male_comparison.png)

- CSV: `results/BlondHair_Male.csv`

### Mouth_Slightly_Open × Smiling

![MouthSlightlyOpen vs Smiling comparison](results/mouth_smiling_comparison.png)

- CSV: `results/MouthSlightlyOpen_Smiling.csv`

### Smiling × High_Cheekbones

- CSV: `results/Smiling_High_Cheekbones.csv`

### Young × Gray_Hair

![Young vs GrayHair comparison](results/young_gray_comparison.png)

- CSV: `results/Young_GrayHair.csv`
