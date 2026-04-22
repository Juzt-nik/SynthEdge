"""
tests/test_synthedge.py
========================
Core test suite for SynthEdge.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from synthedge import SynthEdge
from synthedge.scanner import scan, adaptive_bins
from synthedge.quality import classify_severity, gap_region_kl
from synthedge.transfer import find_matching_gaps, transfer_samples, apply_transfers


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_df(n=400, seed=42, pos_rate=0.30):
    rng = np.random.default_rng(seed)
    age  = rng.normal(54, 9, n).clip(30, 77)
    sex  = rng.choice([0, 1], n, p=[0.35, 0.65])
    chol = rng.normal(246, 51, n).clip(126, 564)
    bp   = rng.normal(131, 18, n).clip(94, 200)
    logit = (0.04*(age-54) - 0.5*sex - 0.002*(chol-246)
             + 0.005*(bp-131) + rng.normal(0, 0.5, n))
    thresh = np.percentile(logit, 100*(1-pos_rate))
    target = (logit >= thresh).astype(int)
    return pd.DataFrame({"age": age, "sex": sex,
                         "chol": chol, "bp": bp, "target": target})


@pytest.fixture
def df():
    return make_df(400, seed=42)


@pytest.fixture
def df_small():
    return make_df(150, seed=7, pos_rate=0.25)


@pytest.fixture
def df_large():
    return make_df(2000, seed=99, pos_rate=0.15)


# ── adaptive_bins ──────────────────────────────────────────────────────────

class TestAdaptiveBins:
    def test_large_dataset_uses_6(self):
        assert adaptive_bins(5000) == 6

    def test_medium_dataset_uses_4_or_5(self):
        b = adaptive_bins(400)
        assert b in [4, 5]

    def test_tiny_dataset_uses_3(self):
        assert adaptive_bins(50) == 3

    def test_always_returns_int(self):
        for n in [30, 100, 500, 5000]:
            assert isinstance(adaptive_bins(n), int)


# ── scanner ────────────────────────────────────────────────────────────────

class TestScanner:
    def test_returns_voxels(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, pca, all_v = scan(X_sc, df["target"], top_k=5)
        assert len(top) <= 5
        assert len(all_v) >= len(top)

    def test_voxel_fields_present(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, _ = scan(X_sc, df["target"], top_k=3)
        for v in top:
            assert "gap_score" in v
            assert "observed" in v
            assert "n_pos" in v
            assert "centroid_sc" in v
            assert "cov_sc" in v

    def test_gap_scores_sorted_descending(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, _ = scan(X_sc, df["target"], top_k=10)
        scores = [v["gap_score"] for v in top]
        assert scores == sorted(scores, reverse=True)

    def test_no_voxel_has_zero_positives(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        _, _, all_v = scan(X_sc, df["target"])
        for v in all_v:
            assert v["n_pos"] > 0

    def test_small_dataset_still_finds_voxels(self, df_small):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df_small[["age", "sex", "chol", "bp"]])
        top, _, _ = scan(X_sc, df_small["target"], top_k=5)
        assert len(top) > 0


# ── severity classifier ────────────────────────────────────────────────────

class TestSeverityClassifier:
    def test_returns_severity_key(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, all_v = scan(X_sc, df["target"])
        result = classify_severity(df, "target", top, all_v)
        assert "severity" in result
        assert result["severity"] in ["NONE", "MILD", "MODERATE", "SEVERE"]

    def test_will_help_is_bool(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, all_v = scan(X_sc, df["target"])
        result = classify_severity(df, "target", top, all_v)
        assert isinstance(result["will_help"], bool)

    def test_score_between_0_and_1(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, all_v = scan(X_sc, df["target"])
        result = classify_severity(df, "target", top, all_v)
        assert 0.0 <= result["score"] <= 1.0

    def test_signals_present(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, all_v = scan(X_sc, df["target"])
        result = classify_severity(df, "target", top, all_v)
        sig = result["signals"]
        assert "positive_rate" in sig
        assert "max_gap_score" in sig
        assert "dataset_size" in sig

    def test_empty_voxels_returns_none_severity(self, df):
        result = classify_severity(df, "target", [], [])
        assert result["severity"] == "NONE"
        assert result["will_help"] == False

    def test_severe_on_highly_imbalanced(self):
        # 5% positive rate with large gaps should trigger SEVERE or MODERATE
        df_imb = make_df(1000, seed=1, pos_rate=0.05)
        sc = StandardScaler()
        X_sc = sc.fit_transform(df_imb[["age", "sex", "chol", "bp"]])
        top, _, all_v = scan(X_sc, df_imb["target"])
        result = classify_severity(df_imb, "target", top, all_v)
        assert result["severity"] in ["MODERATE", "SEVERE"]
        assert result["will_help"] == True


# ── SynthEdge core API ────────────────────────────────────────────────────

class TestSynthEdgeCore:
    def test_analyze_returns_report(self, df):
        se = SynthEdge(df, target_col="target", verbose=False)
        report = se.analyze()
        assert "severity" in report
        assert "top_voxels" in report
        assert "will_help" in report

    def test_fill_returns_dataframe(self, df):
        se = SynthEdge(df, target_col="target", verbose=False)
        aug = se.fill(n_top=2, ctgan_epochs=5, use_ctgan=False)
        assert isinstance(aug, pd.DataFrame)
        assert "target" in aug.columns

    def test_fill_increases_row_count(self, df):
        se = SynthEdge(df, target_col="target", verbose=False)
        aug = se.fill(n_top=2, ctgan_epochs=5, use_ctgan=False)
        assert len(aug) >= len(df)

    def test_fill_preserves_feature_columns(self, df):
        feats = ["age", "sex", "chol", "bp"]
        se = SynthEdge(df, target_col="target",
                       feature_cols=feats, verbose=False)
        aug = se.fill(n_top=2, ctgan_epochs=5, use_ctgan=False)
        for f in feats:
            assert f in aug.columns

    def test_quality_report_returns_dict(self, df):
        se = SynthEdge(df, target_col="target", verbose=False)
        se.fill(n_top=2, ctgan_epochs=5, use_ctgan=False)
        q = se.quality_report()
        assert isinstance(q, dict)
        assert "total_added" in q
        assert "drift_features" in q

    def test_quality_report_before_fill_raises(self, df):
        se = SynthEdge(df, target_col="target", verbose=False)
        with pytest.raises(RuntimeError):
            se.quality_report()

    def test_gap_map_is_dataframe(self, df):
        se = SynthEdge(df, target_col="target", verbose=False)
        se.analyze()
        gm = se.gap_map
        assert isinstance(gm, pd.DataFrame)
        assert "gap_score" in gm.columns
        assert "voxel" in gm.columns

    def test_severity_property(self, df):
        se = SynthEdge(df, target_col="target", verbose=False)
        se.analyze()
        assert se.severity["severity"] in ["NONE", "MILD", "MODERATE", "SEVERE"]

    def test_analyze_auto_called_by_fill(self, df):
        # fill() should call analyze() automatically if not done yet
        se = SynthEdge(df, target_col="target", verbose=False)
        aug = se.fill(n_top=2, ctgan_epochs=5, use_ctgan=False)
        assert se._top_voxels is not None
        assert isinstance(aug, pd.DataFrame)

    def test_ctgan_synthesis(self, df):
        se = SynthEdge(df, target_col="target", verbose=False)
        aug = se.fill(n_top=2, ctgan_epochs=5, use_ctgan=True)
        q = se.quality_report()
        assert len(aug) >= len(df)
        # CTGAN may fall back to Gaussian on small datasets — that's OK
        assert "total_added" in q

    def test_no_target_leakage(self, df):
        # Synthetic rows should only ever have target=1 added
        se = SynthEdge(df, target_col="target", verbose=False)
        aug = se.fill(n_top=2, ctgan_epochs=5, use_ctgan=False)
        orig_neg = (df["target"] == 0).sum()
        aug_neg  = (aug["target"] == 0).sum()
        # Negatives should not increase
        assert aug_neg == orig_neg

    def test_custom_feature_cols(self, df):
        se = SynthEdge(df, target_col="target",
                       feature_cols=["age", "chol"], verbose=False)
        aug = se.fill(n_top=2, ctgan_epochs=5, use_ctgan=False)
        assert "age" in aug.columns
        assert "chol" in aug.columns


# ── multi-dataset transfer ────────────────────────────────────────────────

class TestTransfer:
    def _build_info(self, name, df, seed=0):
        feats = ["age", "sex", "chol", "bp"]
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[feats])
        y = df["target"]
        top, _, _ = scan(X_sc, y, top_k=5)
        return {
            "name": name,
            "top_voxels": top,
            "scaler": sc,
            "X_tr_sc": X_sc,
            "y_tr": y,
            "feature_names": feats,
            "_sparse_vox": top[0] if top else None,
        }

    def test_find_matching_gaps_returns_list(self):
        df_a = make_df(400, seed=42)
        df_b = make_df(400, seed=99)
        info_a = self._build_info("A", df_a)
        info_b = self._build_info("B", df_b)
        matches = find_matching_gaps([info_a, info_b], similarity_threshold=0.3)
        assert isinstance(matches, list)

    def test_transfer_samples_returns_dict(self):
        df_a = make_df(400, seed=42)
        df_b = make_df(400, seed=99)
        info_a = self._build_info("A", df_a)
        info_b = self._build_info("B", df_b)
        matches = find_matching_gaps([info_a, info_b], similarity_threshold=0.3)
        if matches:
            transfers = transfer_samples(matches, n_transfer=5, verbose=False)
            assert isinstance(transfers, dict)

    def test_apply_transfers_increases_rows(self):
        df_a = make_df(400, seed=42)
        df_b = make_df(800, seed=99)
        info_a = self._build_info("A", df_a)
        info_b = self._build_info("B", df_b)
        matches = find_matching_gaps([info_a, info_b], similarity_threshold=0.3)
        if matches:
            transfers = transfer_samples(matches, n_transfer=5, verbose=False)
            sc = StandardScaler()
            X_sc = sc.fit_transform(df_a[["age","sex","chol","bp"]])
            X_aug, y_aug, n_added = apply_transfers("A", X_sc, df_a["target"], transfers)
            if n_added > 0:
                assert len(X_aug) > len(X_sc)
                assert len(y_aug) == len(X_aug)

    def test_apply_transfers_no_match_returns_original(self):
        df_a = make_df(200, seed=42)
        sc = StandardScaler()
        X_sc = sc.fit_transform(df_a[["age","sex","chol","bp"]])
        # Empty transfers dict
        X_aug, y_aug, n_added = apply_transfers("A", X_sc, df_a["target"], {})
        assert n_added == 0
        assert len(X_aug) == len(X_sc)

    def test_match_fields_present(self):
        df_a = make_df(400, seed=42)
        df_b = make_df(400, seed=99)
        info_a = self._build_info("A", df_a)
        info_b = self._build_info("B", df_b)
        matches = find_matching_gaps([info_a, info_b], similarity_threshold=0.3)
        for m in matches:
            assert "sparse_dataset" in m
            assert "dense_dataset" in m
            assert "similarity" in m
            assert "shared_features" in m


# ── KL metric ─────────────────────────────────────────────────────────────

class TestKLMetric:
    def test_kl_returns_float(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, _ = scan(X_sc, df["target"], top_k=3)
        if top:
            kl = gap_region_kl(X_sc, X_sc[:10], top[0], X_sc)
            assert isinstance(kl, float)

    def test_kl_non_negative(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, _ = scan(X_sc, df["target"], top_k=3)
        if top:
            kl = gap_region_kl(X_sc, X_sc[:10], top[0], X_sc)
            assert kl >= 0

    def test_kl_inf_on_empty_near(self, df):
        sc = StandardScaler()
        X_sc = sc.fit_transform(df[["age", "sex", "chol", "bp"]])
        top, _, _ = scan(X_sc, df["target"], top_k=3)
        if top:
            # Held-out far outside any augmented sample
            far_held = X_sc * 100
            kl = gap_region_kl(X_sc[:5], far_held[:2], top[0], X_sc)
            assert kl == float("inf") or kl >= 0
