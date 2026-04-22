"""
synthedge.core
==============
Main SynthEdge class — the user-facing API.

Usage
-----
    from synthedge import SynthEdge

    se = SynthEdge(df, target_col="target")
    report = se.analyze()          # severity + gap map
    aug_df = se.fill(n=500)        # targeted augmentation
    print(se.quality_report())     # KL + drift metrics
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .scanner    import scan
from .synthesizer import synthesize
from .quality    import classify_severity, gap_region_kl, feature_drift, print_severity


class SynthEdge:
    """
    Diagnosis-first synthetic data augmentation.

    Parameters
    ----------
    df           : pd.DataFrame
    target_col   : str — name of the binary target column
    feature_cols : list of str or None — features to use (all non-target if None)
    discrete_cols: list of str or None — categorical features for CTGAN
    verbose      : bool
    """

    def __init__(self, df, target_col="target",
                 feature_cols=None, discrete_cols=None,
                 verbose=True):
        self.df          = df.copy()
        self.target_col  = target_col
        self.verbose     = verbose

        self.feature_cols = (feature_cols if feature_cols is not None
                             else [c for c in df.columns if c != target_col])
        self.discrete_cols = discrete_cols or []

        self.scaler      = StandardScaler()
        self._X_df       = self.df[self.feature_cols]
        self._y          = self.df[self.target_col]
        self._X_sc       = self.scaler.fit_transform(self._X_df)

        # Set after analyze()
        self._top_voxels = None
        self._pca        = None
        self._all_voxels = None
        self._severity   = None

        # Set after fill()
        self._X_aug_sc   = None
        self._y_aug      = None
        self._aug_meta   = None

    # ── analyze ────────────────────────────────────────────────────────────

    def analyze(self, n_bins=None, top_k=10):
        """
        Run gap detection and severity classification.

        Returns
        -------
        dict with keys: severity, score, signals, recommendation,
                        will_help, top_voxels, all_voxels
        """
        if self.verbose:
            print("[SynthEdge] Running 3D local density scan...")

        top_voxels, pca, all_voxels = scan(
            self._X_sc, self._y,
            n_bins=n_bins, top_k=top_k
        )
        self._top_voxels = top_voxels
        self._pca        = pca
        self._all_voxels = all_voxels

        severity = classify_severity(
            self.df, self.target_col, top_voxels, all_voxels
        )
        self._severity = severity

        if self.verbose:
            print_severity(severity)
            n_bins_used = top_voxels[0]["n_bins"] if top_voxels else "?"
            print("[SynthEdge] " + str(len(top_voxels)) + " gap voxels found "
                  "(grid=" + str(n_bins_used) + "^3, "
                  + str(len(all_voxels)) + " non-empty voxels total)")
            print()
            print("  Top 5 gap voxels:")
            for v in top_voxels[:5]:
                print("    " + v["label"] + "  obs=" + str(v["observed"]) +
                      "  exp=" + str(v["expected"]) + "  pos=" + str(v["n_pos"]) +
                      "  sparsity=" + str(v["sparsity"]) + "  score=" + str(v["gap_score"]))
            print()

        return {**severity, "top_voxels": top_voxels, "all_voxels": all_voxels}

    # ── fill ───────────────────────────────────────────────────────────────

    def fill(self, n_top=3, ctgan_epochs=100, use_ctgan=True):
        """
        Synthesize samples targeted at the top gap voxels.

        Parameters
        ----------
        n_top        : int — number of top voxels to synthesize for
        ctgan_epochs : int — CTGAN training epochs
        use_ctgan    : bool — use CTGAN (True) or Gaussian only (False)

        Returns
        -------
        aug_df : pd.DataFrame — augmented dataset (original + synthetic rows)
        """
        if self._top_voxels is None:
            self.analyze()

        if self._severity and not self._severity["will_help"] and self.verbose:
            print("[SynthEdge] WARNING: Severity is " + self._severity["severity"] +
                  ". Augmentation may not improve your model.")
            print("            Proceeding anyway — check quality_report() after filling.")
            print()

        if self.verbose:
            print("[SynthEdge] Synthesizing targeted samples...")

        X_aug, y_aug, meta = synthesize(
            top_voxels    = self._top_voxels,
            X_tr_sc       = self._X_sc,
            y_tr          = self._y,
            X_tr_df       = self._X_df,
            feature_names = self.feature_cols,
            discrete_columns = self.discrete_cols,
            n_top         = n_top,
            ctgan_epochs  = ctgan_epochs,
            use_ctgan     = use_ctgan,
            verbose       = self.verbose,
        )

        self._X_aug_sc = X_aug
        self._y_aug    = y_aug
        self._aug_meta = meta

        if self.verbose:
            n_orig = len(self.df)
            n_added = meta["total_added"]
            print()
            print("[SynthEdge] Added " + str(n_added) + " synthetic positives "
                  "(" + str(n_orig) + " -> " + str(n_orig + n_added) + " rows)")
            method = "CTGAN" if meta.get("ctgan_used") else "Gaussian"
            print("[SynthEdge] Synthesis method: " + method)
            print()

        # Inverse-transform augmented scaled matrix back to original feature space
        X_aug_raw = self.scaler.inverse_transform(X_aug)
        aug_df = pd.DataFrame(X_aug_raw, columns=self.feature_cols)
        aug_df[self.target_col] = y_aug
        return aug_df

    # ── quality_report ─────────────────────────────────────────────────────

    def quality_report(self, held_sc=None):
        """
        Return a dict of quality metrics for the augmentation.

        Parameters
        ----------
        held_sc : np.ndarray or None
            Scaled held-out gap samples (ground truth).
            If None, KL metric is skipped.

        Returns
        -------
        dict with keys: kl, drift_features, total_added, ctgan_used
        """
        if self._X_aug_sc is None:
            raise RuntimeError("Call .fill() before .quality_report()")

        report = {
            "total_added": self._aug_meta["total_added"],
            "ctgan_used":  self._aug_meta.get("ctgan_used", False),
            "voxels":      self._aug_meta.get("voxels", []),
            "kl":          None,
            "drift_features": [],
        }

        if held_sc is not None and self._top_voxels:
            kl = gap_region_kl(
                self._X_aug_sc, held_sc,
                self._top_voxels[0], self._X_sc
            )
            report["kl"] = kl
            if self.verbose:
                print("[SynthEdge] KL divergence in gap region: " + str(kl))

        drifted = feature_drift(self._X_sc, self._X_aug_sc)
        report["drift_features"] = drifted
        if self.verbose and drifted:
            print("[SynthEdge] Feature drift detected in " + str(len(drifted)) +
                  " feature(s):")
            for d in drifted[:5]:
                print("  feature[" + str(d["feature_idx"]) + "] "
                      "ks=" + str(d["ks_stat"]) + " p=" + str(d["p_value"]))

        return report

    # ── convenience properties ─────────────────────────────────────────────

    @property
    def gap_map(self):
        """Return the top voxels as a clean DataFrame."""
        if self._top_voxels is None:
            self.analyze()
        rows = []
        for v in self._top_voxels:
            rows.append({
                "voxel":    v["label"],
                "observed": v["observed"],
                "expected": v["expected"],
                "n_pos":    v["n_pos"],
                "sparsity": v["sparsity"],
                "entropy":  v["entropy"],
                "gap_score":v["gap_score"],
            })
        return pd.DataFrame(rows)

    @property
    def severity(self):
        """Return severity classification result."""
        if self._severity is None:
            self.analyze()
        return self._severity
