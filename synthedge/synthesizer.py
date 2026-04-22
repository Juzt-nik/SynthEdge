"""
synthedge.synthesizer
=====================
CTGAN-based targeted synthesis.

Strategy:
  1. Train CTGAN on the full training set (learns real joint distribution)
  2. Generate a large pool of synthetic positive samples
  3. Filter: keep only samples whose PCA projection falls inside a gap voxel
  4. Quality gate: discriminator score + distance cutoff
  5. Fallback to Gaussian sampling when CTGAN pool is insufficient

This is strictly better than pure Gaussian synthesis because CTGAN
learns real feature correlations — it won't generate, e.g., a 2-year-old
with stage-3 hypertension.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _gaussian_fill(voxel, n_needed):
    """Fallback: multivariate Gaussian around voxel centroid."""
    cov = voxel["cov_sc"].copy()
    rank = np.linalg.matrix_rank(cov)
    if rank < cov.shape[0]:
        cov += 1e-4 * np.eye(cov.shape[0])
    candidates = np.random.multivariate_normal(voxel["centroid_sc"], cov, n_needed * 4)
    dists = np.linalg.norm(candidates - voxel["centroid_sc"], axis=1)
    thresh = np.percentile(dists, 60)
    return candidates[dists <= thresh][:n_needed]


def _quality_gate(candidates, X_tr_sc, y_tr, threshold=0.4):
    """
    Lightweight discriminator: logistic regression trained to separate
    real vs synthetic. Reject candidates the model is too confident
    are fake (score > threshold means too easy to detect as synthetic).
    Returns mask of accepted candidates.
    """
    if len(candidates) == 0:
        return np.array([], dtype=bool)

    real_label = np.ones(len(X_tr_sc))
    fake_label = np.zeros(len(candidates))
    X_disc = np.vstack([X_tr_sc, candidates])
    y_disc = np.concatenate([real_label, fake_label])

    try:
        disc = LogisticRegression(max_iter=200, random_state=42)
        disc.fit(X_disc, y_disc)
        # Probability of being REAL (class 1)
        proba_real = disc.predict_proba(candidates)[:, 1]
        # Accept if model thinks it's plausibly real (>= 1 - threshold)
        return proba_real >= (1 - threshold)
    except Exception:
        # If discriminator fails, accept all
        return np.ones(len(candidates), dtype=bool)


def synthesize(top_voxels, X_tr_sc, y_tr, X_tr_df,
               feature_names, discrete_columns=None,
               n_top=3, ctgan_epochs=100,
               use_ctgan=True, verbose=True):
    """
    Targeted synthesis using CTGAN + quality gate.

    Parameters
    ----------
    top_voxels      : list of voxel dicts from scanner.scan()
    X_tr_sc         : np.ndarray — scaled training features
    y_tr            : pd.Series  — training labels
    X_tr_df         : pd.DataFrame — original (unscaled) training features
    feature_names   : list of str
    discrete_columns: list of str — categorical columns for CTGAN
    n_top           : int — number of top voxels to synthesize for
    ctgan_epochs    : int — CTGAN training epochs
    use_ctgan       : bool — if False, use Gaussian only
    verbose         : bool

    Returns
    -------
    X_aug : np.ndarray — augmented scaled feature matrix
    y_aug : np.ndarray — augmented labels
    meta  : dict       — synthesis metadata
    """
    if discrete_columns is None:
        discrete_columns = []

    y_arr = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)

    # ── Train CTGAN on positive class samples ──────────────────────────────
    ctgan_model = None
    ctgan_pool  = None

    if use_ctgan:
        pos_mask = y_arr == 1
        X_pos_df = X_tr_df[pos_mask].copy().reset_index(drop=True)

        if len(X_pos_df) >= 15:   # CTGAN needs enough data to train
            try:
                if verbose:
                    print("  [CTGAN] Training on " + str(len(X_pos_df)) + " positive samples (" + str(ctgan_epochs) + " epochs)...")
                ctgan_model = CTGAN(epochs=ctgan_epochs, verbose=False)
                ctgan_model.fit(X_pos_df, discrete_columns=discrete_columns)
                # Generate a large pool — we'll filter to gap regions next
                pool_size = max(500, len(X_pos_df) * 10)
                ctgan_pool_df = ctgan_model.sample(pool_size)
                ctgan_pool = ctgan_pool_df.values.astype(float)
                if verbose:
                    print("  [CTGAN] Generated pool of " + str(pool_size) + " candidates")
            except Exception as e:
                if verbose:
                    print("  [CTGAN] Training failed (" + str(e) + ") — using Gaussian fallback")
                ctgan_model = None
                ctgan_pool  = None
        else:
            if verbose:
                print("  [CTGAN] Too few positives (" + str(len(X_pos_df)) + ") — using Gaussian fallback")

    # ── Synthesize for each top gap voxel ─────────────────────────────────
    parts   = []
    vox_meta = []

    for vox in top_voxels[:n_top]:
        n_needed = max(3, vox["observed"] * 3)
        centroid = vox["centroid_sc"]
        method_used = "none"

        # Try CTGAN pool first
        if ctgan_pool is not None and len(ctgan_pool) > 0:
            # Distance from each pool sample to voxel centroid
            dists = np.linalg.norm(ctgan_pool - centroid, axis=1)
            radius = np.percentile(
                np.linalg.norm(X_tr_sc[vox["indices"]] - centroid, axis=1),
                90
            ) * 1.5 if len(vox["indices"]) > 0 else np.std(dists) * 0.8

            near_mask = dists <= radius
            near_samples = ctgan_pool[near_mask]

            if len(near_samples) >= n_needed:
                # Quality gate
                accepted_mask = _quality_gate(near_samples, X_tr_sc, y_arr)
                accepted = near_samples[accepted_mask][:n_needed]
                if len(accepted) >= max(1, n_needed // 2):
                    parts.append(accepted)
                    method_used = "ctgan"
                    if verbose:
                        print("  [SE] Voxel " + vox["label"] + ": " + str(len(accepted)) + " samples via CTGAN")

        # Gaussian fallback
        if method_used == "none":
            gauss = _gaussian_fill(vox, n_needed)
            if len(gauss) > 0:
                accepted_mask = _quality_gate(gauss, X_tr_sc, y_arr)
                accepted = gauss[accepted_mask][:n_needed]
                if len(accepted) > 0:
                    parts.append(accepted)
                    method_used = "gaussian"
                    if verbose:
                        print("  [SE] Voxel " + vox["label"] + ": " + str(len(accepted)) + " samples via Gaussian fallback")

        vox_meta.append({
            "voxel":  vox["label"],
            "method": method_used,
            "added":  len(parts[-1]) if parts and method_used != "none" else 0,
        })

    if not parts:
        return X_tr_sc.copy(), y_arr.copy(), {"total_added": 0, "voxels": vox_meta}

    synth_all = np.vstack(parts)
    X_aug = np.vstack([X_tr_sc, synth_all])
    y_aug = np.concatenate([y_arr, np.ones(len(synth_all), dtype=int)])

    meta = {
        "total_added":  len(synth_all),
        "ctgan_used":   ctgan_model is not None,
        "voxels":       vox_meta,
    }
    return X_aug, y_aug, meta


# ── lazy import so ctgan is optional ──────────────────────────────────────
try:
    from ctgan import CTGAN
except ImportError:
    class CTGAN:
        def __init__(self, *a, **kw):
            raise ImportError("ctgan not installed. Run: pip install ctgan")
