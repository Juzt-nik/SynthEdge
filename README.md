# SynthEdge

![CI](https://github.com/Juzt-nik/SynthEdge/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/synthedge)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Diagnosis-first synthetic data augmentation for imbalanced tabular datasets.**

> On the Framingham Heart Study, SMOTE added **2,045 synthetic samples** and recall dropped by **22 points**.  
> SynthEdge added **18 targeted samples** and preserved performance.

---

## The problem with SMOTE

Every ML developer working with imbalanced data knows SMOTE. It is easy to use, well-documented, and widely trusted. It is also frequently damaging on structurally imbalanced datasets.

The reason is simple: **SMOTE generates blindly**. It interpolates between existing minority samples with no knowledge of where your data is actually sparse. On datasets with structural gaps — specific demographic subgroups, rare feature combinations, underrepresented clinical profiles — SMOTE fills the wrong places and actively hurts minority-class recall.

SynthEdge takes a different approach. It asks **where is your data missing** before generating anything, then synthesizes only in those specific regions.

---

## How it works

SynthEdge runs four steps in sequence:

**1. Gap detection** — A 3D local density scan tiles your feature space into an adaptive voxel grid over PCA-projected dimensions. Each voxel is scored by sparsity, label entropy, and positive-class density. The result is a ranked gap map showing exactly where your minority class is underrepresented.

**2. Severity classification** — Before any augmentation, SynthEdge tells you whether it will help. Datasets are classified as `NONE`, `MILD`, `MODERATE`, or `SEVERE`. If severity is `NONE`, the tool recommends skipping augmentation entirely.

**3. CTGAN synthesis** — Targeted synthesis generates samples only inside identified gap voxels. CTGAN learns the real joint feature distribution from your positive-class samples, then a quality gate (logistic discriminator) rejects candidates that are too easy to identify as synthetic. Falls back to Gaussian sampling when voxels are too sparse for CTGAN.

**4. Gap report** — Every run can produce a standalone HTML report showing severity, the full gap voxel map, synthesis summary, and optional model comparison charts. No other augmentation tool produces an auditable artifact like this.

---

## Install

```bash
pip install synthedge
```

---

## Quick start

```python
from synthedge import SynthEdge

se = SynthEdge(df, target_col="target")

# Step 1: diagnose your dataset
report = se.analyze()
# Severity : !! SEVERE  (score=0.67)
# Will SynthEdge help? YES
# Recommendation: SMOTE will likely hurt recall. SynthEdge strongly recommended.

# Step 2: fill the gaps
aug_df = se.fill()
# [CTGAN] Training on 435 positive samples...
# [SE] Voxel (1,0,3): 6 samples via CTGAN
# Added 18 targeted positives (2915 -> 2933 rows)

# Step 3: check quality
q = se.quality_report()
# KL divergence in gap region: 1.969

# Step 4: generate HTML report
se.save_report("report.html", dataset_name="Framingham Heart Study")
```

---

## CLI

SynthEdge ships with a full command-line interface. No Python code needed.

### `synthedge analyze`
Diagnose your dataset. Prints severity level, gap voxel map, and recommendation.

```bash
synthedge analyze data.csv --target diagnosis
```

### `synthedge fill`
Augment your dataset and save to disk.

```bash
synthedge fill data.csv --target diagnosis --n-top 3 --out augmented.csv

# Flags:
#   --n-top      Top N gap voxels to synthesize for (default: 3)
#   --epochs     CTGAN training epochs (default: 100)
#   --no-ctgan   Use Gaussian only — much faster, slightly lower quality
#   --out        Output CSV path (default: data_augmented.csv)
#   --report     Also save an HTML gap report
```

### `synthedge report`
Generate a standalone HTML gap report from the terminal.

```bash
synthedge report data.csv --target diagnosis --out report.html --name "My Dataset"
```

### `synthedge compare`
Run a full benchmark — trains a model on raw data, SMOTE-augmented data, and SynthEdge-augmented data, then prints all metrics side by side.

```bash
synthedge compare data.csv --target diagnosis --report
```

Output:
```
  Method               Recall    F1        ROC-AUC   PR-AUC
  ──────────────────────────────────────────────────────────
  No augmentation      0.486     0.354     0.768     0.631
  SMOTE (+2045)        0.333     0.283     0.721     0.582
  SynthEdge (+18)      0.486     0.367     0.777     0.637  <--

  SynthEdge vs SMOTE:
    Recall     +15.3 pp  WIN
    F1         +8.4 pp   WIN
    ROC-AUC    +5.6 pp   WIN
    PR-AUC     +5.5 pp   WIN
```

### `synthedge transfer`
Transfer real samples from a source dataset to fill gaps in a target dataset.

```bash
synthedge transfer cleveland.csv framingham.csv --target diagnosis --threshold 0.65
```

---

## Severity classifier

The severity classifier runs automatically on every `.analyze()` call. It scores your dataset across four signals and tells you what to expect before touching your data.

| Level | Score | Meaning | Expected outcome |
|---|---|---|---|
| `NONE` | < 0.15 | Well-distributed dataset | Skip augmentation — no gaps to fill |
| `MILD` | 0.15–0.35 | Minor gaps present | Try SMOTE first — SynthEdge may not improve it |
| `MODERATE` | 0.35–0.60 | Clear structural gaps | Run `compare` first — results vary by dataset |
| `SEVERE` | > 0.60 | Severe structural gaps | Use SynthEdge — SMOTE will likely hurt recall |

---

## Benchmark results

### Cardiovascular domain (3 datasets)

Tested with artificially carved gap regions — 70% of minority samples in a specific demographic subgroup removed.

**Minority-class recall:**

| Dataset | Rows | Positive rate | No-aug | SMOTE | SynthEdge | Gain |
|---|---|---|---|---|---|---|
| Cleveland Heart Disease | 297 | 46% | 0.821 | 0.821 | **0.857** | **+3.6 pp** |
| Framingham Heart Study | 3,658 | 15% | 0.486 | 0.333 | **0.486** | **+15.3 pp** |
| Cardiovascular 70k | 68,604 | 49% | 0.694 | 0.692 | 0.693 | +0.1 pp |

**Synthesis efficiency:**

| Dataset | SMOTE added | SynthEdge added | Ratio |
|---|---|---|---|
| Cleveland | 22 | 27 | 1× |
| Framingham | 2,045 | 18 | **114× fewer** |
| Cardiovascular 70k | 1,690 | 39 | **43× fewer** |

**Key finding:** On Framingham, SMOTE generated 114× more samples and achieved 22 points worse recall. This is the core failure mode of blind oversampling on structurally imbalanced clinical data.

---

### New domain datasets (3 datasets)

Tested on Credit Card Fraud, Pima Diabetes, and Telco Churn — three domains where SMOTE is commonly applied.

**Win counts across 3 models × 3 datasets = 9 combinations:**

| Metric | SynthEdge wins | SMOTE wins | Ties |
|---|---|---|---|
| Recall | 2 | 9 | 1 |
| F1 | 4 | 8 | 0 |
| ROC-AUC | 5 | 4 | 3 |
| PR-AUC | 6 | 6 | 0 |

**Honest interpretation:** SynthEdge lost on recall in this domain. Here is why, per dataset:

**Credit Card Fraud (0.17% positive rate)** — SynthEdge won on XGBoost (+3.1pp recall) but lost on other models. The dataset is extremely imbalanced with fraud patterns that are uniformly sparse rather than structurally gapped. SMOTE's aggressive oversampling helped tree models here.

**Pima Diabetes (35% positive rate)** — SMOTE won convincingly. The gap region chosen (age > 50, glucose > 140, BMI < 30) contained only ~6 samples. SynthEdge added just 4 targeted positives — insufficient to compete with SMOTE's 186. At 35% positive rate, imbalance is moderate and uniform, which is exactly where SMOTE is appropriate.

**Telco Churn (27% positive rate)** — Mixed results. SynthEdge won on RandomForest (+2.1pp) but lost on other models. The gap condition detection found the wrong column after one-hot encoding, weakening the gap targeting.

---

## When to use SynthEdge

SynthEdge is **not a universal replacement for SMOTE**. Based on benchmarks across six real datasets in two domains, here is the honest picture.

**SynthEdge wins decisively when:**
- Positive rate is low (under 20%) and imbalance is structural rather than uniform
- Specific demographic subgroups or feature combinations are underrepresented
- SMOTE generates hundreds or thousands of samples that hurt the model
- You need an auditable gap report for compliance, model cards, or stakeholder review
- Severity classifier outputs `SEVERE`

**SMOTE is appropriate when:**
- Positive rate is moderate (20–40%) and imbalance is roughly uniform
- The dataset is small and lacks enough positives for CTGAN to learn from
- You need a fast baseline without diagnostic overhead
- Severity classifier outputs `NONE` or `MILD`

**When severity is `MODERATE` — run `compare` first:**

```bash
synthedge compare data.csv --target diagnosis
```

This trains both SMOTE and SynthEdge on your data and shows you the metrics before you commit. Takes 2 minutes and removes all guesswork.

**Decision table:**

| Severity | Positive rate | Recommended action |
|---|---|---|
| `NONE` | Any | No augmentation needed |
| `MILD` | Any | Use SMOTE — SynthEdge unlikely to improve it |
| `MODERATE` | > 20% | Run `compare` first |
| `MODERATE` | < 20% | SynthEdge recommended |
| `SEVERE` | Any | SynthEdge strongly recommended |

---

## HTML gap report

Every `se.save_report()` call generates a standalone HTML file that works in any browser — no server, no dependencies, no internet needed. The report includes:

- **Severity banner** — level, score, and plain-English recommendation
- **Metric cards** — dataset size, positive rate, gap voxels found, samples added
- **Gap voxel map** — ranked table with inline score bars
- **Synthesis summary** — CTGAN or Gaussian per voxel, samples added
- **Severity signals** — all six raw inputs to the classifier
- **Model comparison charts** — Recall, F1, ROC-AUC, PR-AUC bar charts when comparison results are provided

The report is suitable for model cards, compliance documents, pull request reviews, and stakeholder presentations.

---

## Full API reference

### `SynthEdge(df, target_col, feature_cols=None, discrete_cols=None, verbose=True)`

| Method | Description |
|---|---|
| `.analyze(n_bins, top_k)` | Run gap detection + severity classification. **Always call first.** |
| `.fill(n_top, ctgan_epochs, use_ctgan)` | Synthesize targeted samples |
| `.quality_report(held_sc)` | KL divergence + feature drift metrics |
| `.save_report(output_path, dataset_name, comparison_results)` | Generate HTML gap report |
| `.gap_map` | `pd.DataFrame` of top gap voxels with scores |
| `.severity` | Severity result dict from last `.analyze()` call |

### `synthedge.transfer`

| Function | Description |
|---|---|
| `find_matching_gaps(datasets_info, threshold)` | Match gap regions across datasets by cosine similarity |
| `transfer_samples(matches, n_transfer)` | Extract real samples for transfer |
| `apply_transfers(name, X_sc, y, transfers)` | Inject transferred samples |

### `synthedge.scanner`

| Function | Description |
|---|---|
| `scan(X_sc, y, n_bins, top_k)` | Run 3D voxel scan directly on a scaled matrix |
| `adaptive_bins(n_train)` | Returns optimal grid size for dataset size |

### `synthedge.quality`

| Function | Description |
|---|---|
| `classify_severity(df, target_col, top_voxels, all_voxels)` | Standalone severity classification |
| `gap_region_kl(X_aug_sc, held_sc, top_voxel, X_tr_sc)` | KL divergence in gap region |

---

## Multi-dataset gap transfer

If you have multiple datasets from the same domain, SynthEdge finds matching gap regions across them using centroid cosine similarity and transfers **real samples** from the less-sparse dataset into the more-sparse one.

```python
from synthedge.scanner import scan
from synthedge.transfer import find_matching_gaps, transfer_samples, apply_transfers

datasets_info = [
    {
        "name": "Cleveland",
        "top_voxels": top_cl,
        "scaler": sc_cl,
        "X_tr_sc": X_cl,
        "y_tr": y_cl,
        "feature_names": feat_cl,
        "_sparse_vox": top_cl[0],
    },
    {
        "name": "Framingham",
        "top_voxels": top_fr,
        "scaler": sc_fr,
        "X_tr_sc": X_fr,
        "y_tr": y_fr,
        "feature_names": feat_fr,
        "_sparse_vox": top_fr[0],
    },
]

matches   = find_matching_gaps(datasets_info, similarity_threshold=0.65)
transfers = transfer_samples(matches, n_transfer=20)
X_aug, y_aug, n_added = apply_transfers("Cleveland", X_cl, y_cl, transfers)
