"""
synthedge.scanner
=================
3D local density scan over PCA-projected feature space.
Finds sparse regions (gap voxels) in the minority-class distribution.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import entropy


def adaptive_bins(n_train):
    """
    Pick n_bins so each voxel has at least 5 expected samples.
    Prevents meaningless sparsity scores on tiny datasets.
    """
    for b in [6, 5, 4, 3]:
        if n_train / (b ** 3) >= 5:
            return b
    return 3


def compute_gap_score(obs, expected, n_pos, ent):
    """
    gap_score = 0.5 * sparsity + 0.3 * entropy + 0.2 * pos_rate
    Higher = more underrepresented and worth filling.
    """
    sparsity = max(0.0, 1.0 - obs / expected)
    pos_rate = n_pos / max(obs, 1)
    return round(0.5 * sparsity + 0.3 * ent + 0.2 * pos_rate, 4)


def scan(X_sc, y, n_bins=None, top_k=10, pca_components=3):
    """
    Run 3D local density scan on scaled feature matrix X_sc.

    Parameters
    ----------
    X_sc : np.ndarray, shape (n, p)  — StandardScaler-transformed features
    y    : pd.Series or np.ndarray   — binary target (0/1)
    n_bins : int or None             — voxel grid side length (auto if None)
    top_k  : int                     — number of top gap voxels to return
    pca_components : int             — PCA dims to project into

    Returns
    -------
    top_voxels : list of dicts, sorted by gap_score descending
    pca        : fitted PCA object (for projecting new data)
    all_voxels : full list of scored voxels
    """
    if hasattr(y, "values"):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)

    n_train = len(X_sc)
    if n_bins is None:
        n_bins = adaptive_bins(n_train)

    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    pc1, pc2, pc3 = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]
    e1 = np.linspace(pc1.min() - 0.01, pc1.max() + 0.01, n_bins + 1)
    e2 = np.linspace(pc2.min() - 0.01, pc2.max() + 0.01, n_bins + 1)
    e3 = np.linspace(pc3.min() - 0.01, pc3.max() + 0.01, n_bins + 1)
    expected = n_train / (n_bins ** 3)

    voxels = []
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                mask = (
                    (pc1 >= e1[i]) & (pc1 < e1[i + 1]) &
                    (pc2 >= e2[j]) & (pc2 < e2[j + 1]) &
                    (pc3 >= e3[k]) & (pc3 < e3[k + 1])
                )
                obs = int(mask.sum())
                if obs == 0:
                    continue
                y_v = y_arr[mask]
                n_pos = int(y_v.sum())
                if n_pos == 0:
                    continue

                p = np.array([1 - y_v.mean(), y_v.mean()]) + 1e-10
                p /= p.sum()
                ent = float(entropy(p, base=2))
                gap_score = compute_gap_score(obs, expected, n_pos, ent)

                cell_X = X_sc[mask]
                cov = np.cov(cell_X.T) if cell_X.shape[0] > 1 else np.eye(cell_X.shape[1])
                cov = cov + 1e-5 * np.eye(cell_X.shape[1])

                voxels.append({
                    "label":        "(" + str(i) + "," + str(j) + "," + str(k) + ")",
                    "i": i, "j": j, "k": k,
                    "observed":     obs,
                    "expected":     round(expected, 2),
                    "n_pos":        n_pos,
                    "sparsity":     round(max(0.0, 1.0 - obs / expected), 3),
                    "entropy":      round(ent, 3),
                    "gap_score":    gap_score,
                    "centroid_sc":  cell_X.mean(axis=0),
                    "centroid_pca": X_pca[mask].mean(axis=0),
                    "cov_sc":       cov,
                    "indices":      np.where(mask)[0],
                    "n_bins":       n_bins,
                    "expected_raw": expected,
                })

    voxels.sort(key=lambda v: v["gap_score"], reverse=True)
    return voxels[:top_k], pca, voxels
