"""
synthedge.quality
=================
Two things:
  1. Severity classifier — tells you upfront whether SynthEdge will help
  2. KL divergence metric — measures how well gap recovery worked
"""

import numpy as np
from scipy.stats import entropy, ks_2samp


# ── Severity classifier ────────────────────────────────────────────────────

SEVERITY_NONE     = "NONE"
SEVERITY_MILD     = "MILD"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_SEVERE   = "SEVERE"


def classify_severity(df, target_col, top_voxels, all_voxels):
    """
    Analyse the gap structure and return a severity level + recommendation.

    Severity is based on four signals:
      - Positive rate (class imbalance)
      - Max gap score across all voxels
      - Fraction of voxels that are "high gap" (score > 0.6)
      - Dataset size (small datasets have noisier gaps)

    Returns
    -------
    dict with keys:
      severity     : str (NONE / MILD / MODERATE / SEVERE)
      score        : float (0–1, aggregate severity)
      signals      : dict of individual signal values
      recommendation : str (plain-English advice)
      will_help    : bool (honest prediction of whether SE beats SMOTE)
    """
    pos_rate   = float(df[target_col].mean())
    n_rows     = len(df)
    n_voxels   = len(all_voxels)

    if n_voxels == 0:
        return {
            "severity": SEVERITY_NONE,
            "score": 0.0,
            "signals": {},
            "recommendation": "No gap voxels found. Dataset appears uniformly distributed.",
            "will_help": False,
        }

    max_gap_score  = top_voxels[0]["gap_score"] if top_voxels else 0.0
    mean_gap_score = np.mean([v["gap_score"] for v in top_voxels])
    high_gap_frac  = sum(1 for v in all_voxels if v["gap_score"] > 0.6) / max(n_voxels, 1)
    imbalance_signal = max(0.0, 1.0 - 2 * pos_rate)   # 0 if balanced, 1 if all-negative
    size_penalty   = min(1.0, 300 / max(n_rows, 1))    # small datasets penalised

    # Aggregate severity score (0–1)
    raw_score = (
        0.35 * max_gap_score +
        0.25 * mean_gap_score +
        0.25 * high_gap_frac +
        0.15 * imbalance_signal
    )
    score = round(float(np.clip(raw_score - 0.3 * size_penalty, 0, 1)), 4)

    # Classify
    if score < 0.15:
        severity   = SEVERITY_NONE
        will_help  = False
        rec = (
            "Your dataset has minimal structural gaps. "
            "Standard augmentation (SMOTE) or no augmentation may be sufficient. "
            "SynthEdge is unlikely to improve over baseline."
        )
    elif score < 0.35:
        severity   = SEVERITY_MILD
        will_help  = True
        rec = (
            "Mild structural gaps detected. SynthEdge will likely match or slightly "
            "improve over SMOTE. Run with default settings."
        )
    elif score < 0.60:
        severity   = SEVERITY_MODERATE
        will_help  = True
        rec = (
            "Moderate structural gaps detected. SynthEdge is recommended over SMOTE. "
            "Expect meaningful recall improvement on the minority class."
        )
    else:
        severity   = SEVERITY_SEVERE
        will_help  = True
        rec = (
            "Severe structural gaps detected. SMOTE will likely hurt recall. "
            "SynthEdge is strongly recommended. "
            "Consider increasing n_top (synthesize from more voxels)."
        )

    return {
        "severity": severity,
        "score":    score,
        "signals": {
            "positive_rate":    round(pos_rate, 4),
            "max_gap_score":    round(max_gap_score, 4),
            "mean_gap_score":   round(mean_gap_score, 4),
            "high_gap_fraction":round(high_gap_frac, 4),
            "imbalance_signal": round(imbalance_signal, 4),
            "dataset_size":     n_rows,
        },
        "recommendation": rec,
        "will_help": will_help,
    }


def print_severity(result):
    """Pretty-print severity report to stdout."""
    sev   = result["severity"]
    score = result["score"]
    icons = {SEVERITY_NONE: "✓", SEVERITY_MILD: "~",
             SEVERITY_MODERATE: "!", SEVERITY_SEVERE: "!!"}
    icon  = icons.get(sev, "?")

    print("\n" + "─" * 55)
    print("  SYNTHEDGE SEVERITY REPORT")
    print("─" * 55)
    print("  Severity : " + icon + " " + sev + "  (score=" + str(score) + ")")
    print()
    sig = result["signals"]
    print("  Positive rate      : " + str(round(sig.get("positive_rate", 0) * 100, 1)) + "%")
    print("  Max gap score      : " + str(sig.get("max_gap_score", 0)))
    print("  Mean gap score     : " + str(sig.get("mean_gap_score", 0)))
    print("  High-gap voxels    : " + str(round(sig.get("high_gap_fraction", 0) * 100, 1)) + "%")
    print("  Dataset size       : " + str(sig.get("dataset_size", 0)) + " rows")
    print()
    print("  Will SynthEdge help? " + ("YES" if result["will_help"] else "UNLIKELY"))
    print()
    print("  Recommendation:")
    for line in result["recommendation"].split(". "):
        if line.strip():
            print("    " + line.strip() + ".")
    print("─" * 55 + "\n")


# ── KL divergence metric ───────────────────────────────────────────────────

def gap_region_kl(X_aug_sc, held_sc, top_voxel, X_tr_sc, bins=10):
    """
    KL divergence between augmented set (near gap centroid) and
    held-out ground-truth gap samples. Lower = better recovery.
    """
    centroid  = top_voxel["centroid_sc"]
    direction = centroid - X_tr_sc.mean(axis=0)
    norm      = np.linalg.norm(direction)
    if norm < 1e-8:
        return float("inf")
    direction /= norm

    proj_aug  = X_aug_sc @ direction
    proj_held = held_sc  @ direction
    mu, sd    = proj_held.mean(), proj_held.std() + 1e-8
    near      = proj_aug[np.abs(proj_aug - mu) < 2.0 * sd]
    if len(near) < 3:
        return float("inf")

    lo = min(proj_held.min(), near.min()) - 0.1
    hi = max(proj_held.max(), near.max()) + 0.1
    p, _ = np.histogram(proj_held, bins=bins, range=(lo, hi), density=True)
    q, _ = np.histogram(near,      bins=bins, range=(lo, hi), density=True)
    p = p + 1e-10;  q = q + 1e-10
    p /= p.sum();   q /= q.sum()
    return round(float(entropy(p, q)), 4)


def feature_drift(X_orig_sc, X_aug_sc, threshold=0.05):
    """
    KS test per feature between original and augmented sets.
    Returns list of features with significant drift (p < threshold).
    """
    drifted = []
    n_feats = X_orig_sc.shape[1]
    for i in range(n_feats):
        stat, p = ks_2samp(X_orig_sc[:, i], X_aug_sc[:, i])
        if p < threshold:
            drifted.append({"feature_idx": i, "ks_stat": round(stat, 4), "p_value": round(p, 4)})
    return drifted
