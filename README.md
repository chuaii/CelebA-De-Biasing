# Fair Classification on CelebA — Q1 Action Plan

## 1. De-biasing Method

**Task:** Predict `Blond_Hair` on CelebA. The **Blond_Male** group is only **0.85 %** of training data, causing ERM to learn a spurious *"blond ≈ female"* shortcut.

**Method:** Fair Supervised Contrastive Learning (FairSupCon) + group-balanced resampling, with progressive ablation to isolate each component's contribution. Group DRO (Sagawa et al., ICLR 2020) as external baseline.

| Stage | Sampling | Loss | Addresses |
|-------|----------|------|-----------|
| **Baseline (ERM)** | Uniform | CE | — (control) |
| **+ Resampling** | Group-balanced | CE | Data imbalance |
| **+ FairSupCon** | Group-balanced | CE + λ · FairSupCon | Feature entanglement |

## 2. Formulation

**Architecture.** ResNet-18 backbone → projection head $\mathbf{z}_i = \text{normalize}(\text{MLP}(\mathbf{h}_i))$ for contrastive loss + classification head for CE loss.

**Positive pairs.** Same target, different sensitive attribute — forcing gender-invariant representations:

$$\mathcal{P}(i) = \left\lbrace j \neq i \mid y_j = y_i \wedge s_j \neq s_i \right\rbrace$$

**FairSupCon loss** ($\tau$ = temperature):

$$\mathcal{L}_{\text{FSC}} = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \frac{1}{|\mathcal{P}(i)|} \sum_{j \in \mathcal{P}(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}$$

**Total objective:**

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{FSC}}$$

## 3. Experiment Design

| Comparison | Purpose |
|------------|---------|
| ERM vs. ERM + Resampling | Quantify effect of balancing alone |
| ERM + Resampling vs. + FairSupCon | Quantify effect of contrastive de-biasing |
| FairSupCon vs. Group DRO | Compare against established de-biasing method |

**Primary metric:** Worst-Group Accuracy (WGA) = $\min_g \text{Acc}_g$. Supplementary: DPD, EOD, Equalized Odds.

## 4. Responsibilities

| Member | Responsibility |
|--------|----------------|
| **Vaibhav** | Baseline ERM; Group DRO baseline; resampling |
| **Huayi** | FairSupCon loss design & implementation; fairness evaluation |
| **Matthew** | FairSupCon integration & tuning; ablation experiments |

## 5. Timeline & Milestones

| Date | Milestone |
|------|-----------|
| **Mar 6 (Q1)** | Problem defined; action plan finalized |
| **Mar 13 (Q2)** | Pipeline working; baseline + FairSupCon initial results |
| **Mar 20** | Hyperparameter sweep; ablation analysis |
| **Mar 27 (Q3)** | Full comparison; fairness evaluation; final report |
