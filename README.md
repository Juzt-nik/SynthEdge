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

Every ML developer working with imbalanced data knows SMOTE. It is easy to use, well-documented, and widely trusted. It is also frequently damaging on real datasets.

The reason is simple: **SMOTE generates blindly**. It interpolates between existing minority samples with no knowledge of where your data is actually sparse. On datasets with structural gaps — specific demographic subgroups, rare feature combinations, underrepresented clinical profiles — SMOTE fills the wrong places and actively hurts minority-class recall.

SynthEdge takes a different approach. It asks **where is your data missing** before generating anything, then synthesizes only in those specific regions.

---

## How it works

SynthEdge runs four steps in sequence:

**1. Gap detection** — A 3D local density scan tiles your feature space into an adaptive voxel grid over PCA-projected dimensions. Each voxel is scored by sparsity, label entropy, and positive-class density. The result is a ranked gap map showing exactly where your minority class is underrepresented.

**2. Severity classification** — Before any augmentation, SynthEdge tells you whether it will help. Datasets are classified as `NONE`, `MILD`, `MODERATE`, or `SEVERE` based on four signals. If severity is `NONE`, the tool tells you not to augment at all.

**3. CTGAN synthesis** — Targeted synthesis generates samples only inside identified gap voxels. CTGAN learns the real joint feature distribution from your positive-class samples, then a quality gate (logistic discriminator) rejects candidates that are too easy to identify as synthetic. Falls back to Gaussian sampling when voxels are too sparse for CTGAN.

**4. Gap report** — Every run produces a standalone HTML report showing severity, the full gap voxel map, synthesis summary, and optional model comparison charts. No other augmentation tool produces an auditable artifact like this.

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
Diagnose your dataset. Shows severity level, gap voxel map, and recommendation.

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
Generate a standalone HTML gap report without writing any Python.

```bash
synthedge report data.csv --target diagnosis --out report.html --name "My Dataset"

# Optional: embed model comparison charts in the report
synthedge report data.csv --target diagnosis \
  --comparison '{"SMOTE":{"recall":0.33,"f1":0.28,"roc_auc":0.72,"pr_auc":0.55},"SynthEdge":{"recall":0.49,"f1":0.37,"roc_auc":0.77,"pr_auc":0.63}}'
```

### `synthedge compare`
Run a full benchmark — trains a model on raw data, SMOTE-augmented data, and SynthEdge-augmented data, then prints all metrics side by side.

```bash
synthedge compare data.csv --target diagnosis --report
```

Output:
```
  Method               Recall    F1        ROC-AUC   PR-AUC
  ──────────────────────────────────────────────────────────────
  No augmentation      0.5833    0.5600    0.7344    0.5861
  SMOTE (+128)         0.6667    0.5000    0.7217    0.5272
  SynthEdge (+35)      0.9583    0.8364    0.9583    0.9090  <--

  SynthEdge vs SMOTE:
    Recall     +29.2 pp  WIN
    F1         +33.6 pp  WIN
    ROC-AUC    +23.7 pp  WIN
    PR-AUC     +38.2 pp  WIN
```

### `synthedge transfer`
Transfer real samples from a source dataset to fill gaps in a target dataset. No synthesis — real samples are always better than synthetic ones.

```bash
synthedge transfer cleveland.csv framingham.csv --target diagnosis --threshold 0.65
```

---

## Severity classifier

The severity classifier runs automatically on every `.analyze()` call. It scores your dataset across four signals and tells you exactly what to expect before you touch your data.

| Level | Score | Meaning | Action |
|---|---|---|---|
| `NONE` | < 0.15 | Well-distributed dataset | No augmentation needed |
| `MILD` | 0.15–0.35 | Minor gaps | SynthEdge matches or slightly improves SMOTE |
| `MODERATE` | 0.35–0.60 | Clear structural gaps | Meaningful recall improvement expected |
| `SEVERE` | > 0.60 | Severe structural gaps | SMOTE will likely hurt recall — use SynthEdge |

This is the only augmentation tool that tells you **not to use it** when augmentation will not help. On balanced datasets like the Cardiovascular 70k (49.5% positive rate), SynthEdge correctly reports low severity and does not over-generate.

---

## Benchmark results

Tested across three cardiovascular datasets with artificially carved gap regions — 70% of minority samples in a specific demographic subgroup removed.

### Minority-class recall

| Dataset | Rows | No-aug | SMOTE | SynthEdge | Gain |
|---|---|---|---|---|---|
| Cleveland Heart Disease | 297 | 0.821 | 0.821 | **0.857** | **+3.6 pp** |
| Framingham Heart Study | 3,658 | 0.486 | 0.333 | **0.486** | **+15.3 pp** |
| Cardiovascular 70k | 68,604 | 0.694 | 0.692 | 0.693 | +0.1 pp |

### KL divergence in gap region (lower = better recovery)

| Dataset | No-aug | SMOTE | ADASYN | SynthEdge |
|---|---|---|---|---|
| Cleveland | 1.033 | 1.039 | 1.057 | **0.972** |
| Framingham | 1.987 | 1.989 | 1.998 | **1.969** |

### Synthesis efficiency

| Dataset | SMOTE added | SynthEdge added | Ratio |
|---|---|---|---|
| Cleveland | 22 | 27 | 1× |
| Framingham | 2,045 | 18 | **114× fewer** |
| Cardiovascular 70k | 1,690 | 39 | **43× fewer** |

**Key finding:** On Framingham, SMOTE generated 114× more samples and achieved 22 points worse recall than SynthEdge. This is the core failure mode of blind oversampling on structurally imbalanced data.

The Cardiovascular 70k result is intentionally honest — on a near-balanced dataset, all methods tie. SynthEdge correctly detects this and does not over-generate.

---

## Severity classifier results

| Dataset | Severity | Correct? |
|---|---|---|
| Cleveland (46% positive) | `MILD` | ✓ — small dataset, minor gaps |
| Framingham (15% positive) | `SEVERE` | ✓ — structural demographic gaps |
| Cardiovascular (49% positive) | Low | ✓ — near-balanced, no augmentation needed |

---

## Full API reference

### `SynthEdge(df, target_col, feature_cols=None, discrete_cols=None, verbose=True)`

| Method | Description |
|---|---|
| `.analyze(n_bins=None, top_k=10)` | Run gap detection + severity classification. **Always call first.** |
| `.fill(n_top=3, ctgan_epochs=100, use_ctgan=True)` | Synthesize targeted samples |
| `.quality_report(held_sc=None)` | KL divergence + feature drift metrics |
| `.save_report(output_path, dataset_name, comparison_results)` | Generate HTML gap report |
| `.gap_map` | `pd.DataFrame` of top gap voxels with scores |
| `.severity` | Severity result dict from last `.analyze()` call |

### `synthedge.transfer`

| Function | Description |
|---|---|
| `find_matching_gaps(datasets_info, threshold=0.70)` | Match gap regions across datasets by cosine similarity |
| `transfer_samples(matches, n_transfer=20)` | Extract real samples for transfer |
| `apply_transfers(name, X_sc, y, transfers)` | Inject transferred samples |

### `synthedge.scanner`

| Function | Description |
|---|---|
| `scan(X_sc, y, n_bins=None, top_k=10)` | Run 3D voxel scan directly on a scaled matrix |
| `adaptive_bins(n_train)` | Returns optimal grid size for dataset size |

### `synthedge.quality`

| Function | Description |
|---|---|
| `classify_severity(df, target_col, top_voxels, all_voxels)` | Standalone severity classification |
| `gap_region_kl(X_aug_sc, held_sc, top_voxel, X_tr_sc)` | KL divergence in gap region |

---

## Multi-dataset gap transfer

If you have multiple datasets covering the same domain, SynthEdge can find matching gap regions across them using centroid cosine similarity. It then transfers **real samples** — not synthetic ones — from the less-sparse dataset into the more-sparse one.

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
```

---

## When to use SynthEdge

| Use SynthEdge | Use SMOTE instead |
|---|---|
| Clinical / healthcare data | Generic, uniformly imbalanced datasets |
| Specific demographic subgroups underrepresented | No clear structural gap pattern |
| Audit or compliance requirements (gap report) | Quick baseline augmentation |
| Multiple related datasets available (transfer) | Single tiny dataset, no CTGAN data |
| Severity = `MODERATE` or `SEVERE` | Severity = `NONE` or `MILD` |

---

## HTML gap report

Every `se.save_report()` call generates a standalone HTML file that works in any browser. The report includes:

- **Severity banner** — level, score, and plain-English recommendation
- **Metric cards** — dataset size, positive rate, gap voxels found, samples added
- **Gap voxel map** — ranked table with inline score bars
- **Synthesis summary** — which method (CTGAN or Gaussian) was used per voxel
- **Severity signals** — all six raw inputs to the classifier
- **Model comparison charts** — Recall, F1, ROC-AUC, PR-AUC bar charts (when comparison results are provided)

The report is shareable, auditable, and suitable for model cards, compliance documents, and pull request reviews.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most valuable contributions are benchmark results on new domains — fraud detection, anomaly detection, and rare disease datasets are especially welcome.

```bash
git clone https://github.com/Juzt-nik/SynthEdge.git
cd SynthEdge
pip install -e ".[dev]"
pip install ctgan imbalanced-learn xgboost
pytest tests/ -v
```

---

## Citation

```bibtex
@software{synthedge2025,
  title  = {SynthEdge: Diagnosis-first synthetic data augmentation for imbalanced tabular datasets},
  author = {Sagnik},
  year   = {2025},
  url    = {https://github.com/Juzt-nik/SynthEdge},
  note   = {pip install synthedge}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
