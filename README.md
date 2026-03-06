# Fair Classification on CelebA — Q1 Action Plan

## 1. De-biasing Method

**Task:** Predict `Blond_Hair` on CelebA. The **Blond_Male** group is only **0.85 %** of training data, causing ERM to learn a spurious *"blond ≈ female"* shortcut.

**Method:** Fair Supervised Contrastive Learning (FairSupCon) + group-balanced resampling, with progressive ablation to isolate each component's contribution. Group DRO (Sagawa et al., ICLR 2020) as external baseline.


| Stage              | Sampling       | Loss                | Addresses            |
| ------------------ | -------------- | ------------------- | -------------------- |
| **Baseline (ERM)** | Uniform        | CE                  | — (control)          |
| **+ Resampling**   | Group-balanced | CE                  | Data imbalance       |
| **+ FairSupCon**   | Group-balanced | CE + λ · FairSupCon | Feature entanglement |


## 2. Formulation

### 2.1 Background — Supervised Contrastive Learning

Supervised Contrastive Learning (SupCon, Khosla et al., NeurIPS 2020) extends self-supervised contrastive learning by using label information to define positive pairs. For anchor $i$, all samples sharing the same label form positive pairs:

$$\mathcal{P}_{\text{SupCon}}(i) = \left\lbrace j \neq i \mid y_j = y_i \right\rbrace$$

$$\mathcal{L}*{\text{SupCon}} = -\frac{1}{|\mathcal{B}|}\sum*{i \in \mathcal{B}} \frac{1}{|\mathcal{P}(i)|} \sum_{j \in \mathcal{P}(i)} \log \frac{\exp(\text{sim}(i,j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(i,k) / \tau)}$$

where $\text{sim}(i,j) = \mathbf{z}_i \cdot \mathbf{z}_j$ is cosine similarity between L2-normalized embeddings. This objective pulls same-class samples together and pushes different-class samples apart in embedding space.

**Problem:** Standard SupCon treats all same-label pairs equally. In a biased dataset like CelebA, Blond samples are overwhelmingly female, so SupCon still clusters by *gender* rather than by *hair color*.

### 2.2 Our Method — FairSupCon Loss

We modify the positive-pair definition to **only** pair samples with the same target label but **different** sensitive attributes:

$$\mathcal{P}_{\text{Fair}}(i) = \left\lbrace i \neq j \mid y_i = y_j \wedge s_i \neq s_j \right\rbrace$$

For example, a BlondFemale ($y=1, s=0$) is only paired with BlondMale ($y=1, s=1$), never with another BlondFemale. This forces the encoder to learn hair-color features that are invariant to gender.

The loss takes the same functional form as SupCon, but with the modified positive set:

$$\mathcal{L}*{\text{FSC}} = - \frac{1}{|\mathcal{B}|} \sum*{i \in \mathcal{B}} \frac{1}{|\mathcal{P}*{\text{Fair}}(i)|} \sum*{j \in \mathcal{P}_{\text{Fair}}(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}*j / \tau)}{\sum*{k \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}$$

where $\mathcal{B}$ is the set of anchors that have at least one valid positive pair in the mini-batch.

**Architecture.** ResNet-18 backbone produces feature $\mathbf{h}_i$, which branches into:

- **Projection head:** $\mathbf{z}_i = \text{normalize}(\text{MLP}(\mathbf{h}*i)) \in \mathbb{R}^{d}$ — used by $\mathcal{L}*{\text{FSC}}$
- **Classification head:** $\hat{y}_i = \text{Linear}(\mathbf{h}*i)$ — used by $\mathcal{L}*{\text{CE}}$

### 2.3 Total Objective & Hyperparameters

$$\mathcal{L}*{\text{total}} = \mathcal{L}*{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{FSC}}$$


| Symbol                 | Meaning                                                                      | Value                |
| ---------------------- | ---------------------------------------------------------------------------- | -------------------- |
| $\lambda$              | Weight of FairSupCon loss                                                    | 0.2 (0 for baseline) |
| $\tau$                 | Temperature (sharpness of similarity distribution)                           | 0.07                 |
| $d$                    | Projection embedding dimension                                               | 128                  |
| $B$                    | Batch size                                                                   | 128                  |
| $lr_{\text{head}}$     | Learning rate for projection + classifier heads                              | 1e-4                 |
| $lr_{\text{backbone}}$ | Learning rate for ResNet-18 backbone (lower to preserve pretrained features) | 1e-5                 |
| $wd$                   | Weight decay                                                                 | 1e-4                 |
| —                      | Epochs                                                                       | 20                   |
| —                      | LR scheduler                                                                 | Cosine Annealing     |


## 3. Experiment Design


| Comparison                        | Purpose                                       |
| --------------------------------- | --------------------------------------------- |
| ERM vs. ERM + Resampling          | Quantify effect of balancing alone            |
| ERM + Resampling vs. + FairSupCon | Quantify effect of contrastive de-biasing     |
| FairSupCon vs. Group DRO          | Compare against established de-biasing method |


**Primary metric:** Worst-Group Accuracy (WGA) = $\min_g \text{Acc}_g$. Supplementary: DPD, EOD, Equalized Odds.

## 4. Responsibilities


| Member      | Responsibility                                               |
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


