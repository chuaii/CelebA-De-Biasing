# Fair Classification on CelebA
## Problem

Predict `Blond_Hair` on CelebA while mitigating gender bias. The **Blond_Male** group is only **0.85%** of training data, causing standard ERM to fail on this minority group. Primary metric: **Worst-Group Accuracy (WGA)**.

| Group | Target | Sensitive | Samples | Proportion |
|-------|--------|-----------|---------|------------|
| NonBlond_Female | Non-Blond | Female | 71,629 | 44.0% |
| NonBlond_Male | Non-Blond | Male | 66,874 | 41.1% |
| Blond_Female | Blond | Female | 22,880 | 14.1% |
| **Blond_Male** | **Blond** | **Male** | **1,387** | **0.85%** |

## De-biasing Method — Progressive Ablation
| | Sampling | Loss | Addresses |
|--|----------|------|-----------|
| **Baseline** | Uniform | CrossEntropy | — (control) |
| **+ Resampling** | Group-balanced | CrossEntropy | Data imbalance |
| **+ FairSupCon** | Group-balanced | CrossEntropy + λ·FairSupCon | Feature entanglement |

Each level builds on the previous. FairSupCon pulls **cross-gender positive pairs** (e.g. Blond_Female ↔ Blond_Male) together in embedding space, forcing the model to decouple gender from hair-color.

Additionally compared against **Group DRO** (Sagawa et al., ICLR 2020) as an external baseline.

## Responsibilities

| Member | Task |
|--------|------|
| `Vaibhav` | Baseline (ERM), external baseline (Group DRO) |
| `Vaibhav` | + Group-balanced resampling |
| `Huayi, Matthew` | + FairSupCon loss |

## Timeline

| Date | Milestone |
|------|-----------|
| **Mar 6 (Q1)** | Problem defined; action plan ready |
| **Mar 13 (Q2)** | Code pipeline working; Baseline results available |
| **Mar 27 (Q3)** | All methods compared; ablation analysis complete |

<!-- ## Usage

```bash
pip install -r requirements.txt
python train.py --mode baseline    # Baseline
python train.py --mode debias      # Resampling + FairSupCon
python train_groupdro.py           # Group DRO
python eval.py --checkpoint checkpoints/best_debias.pt --split test
``` -->
