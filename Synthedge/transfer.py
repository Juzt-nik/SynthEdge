"""
synthedge.transfer
==================
Multi-dataset gap transfer.

Core idea: if dataset A has a severe gap in region R, and dataset B has
real samples in region R (it's less sparse there), those real samples
from B are better synthetic candidates than anything we can generate.

Protocol:
  1. Run gap scan on all datasets independently
  2. Find "matching" gap regions across datasets using centroid similarity
     in PCA space (cosine similarity of gap centroid vectors)
  3. For each matched pair: use samples from the less-sparse dataset
     to fill the gap in the more-sparse dataset
  4. Apply quality gate before injecting
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .quality import feature_drift


def _centroid_similarity(c1, c2):
    """Cosine similarity between two centroid vectors."""
    n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(c1, c2) / (n1 * n2))


def find_matching_gaps(datasets_info, similarity_threshold=0.70):
    """
    Find gap regions that appear across multiple datasets.

    Parameters
    ----------
    datasets_info : list of dicts, each with keys:
        name        : str
        top_voxels  : list of voxel dicts from scanner.scan()
        scaler      : fitted StandardScaler
        X_tr_sc     : np.ndarray
        y_tr        : array-like
        feature_names: list of str

    similarity_threshold : float — cosine similarity to consider voxels "matched"

    Returns
    -------
    matches : list of dicts describing matched gap pairs
    """
    matches = []
    n = len(datasets_info)

    for i in range(n):
        for j in range(i + 1, n):
            ds_a = datasets_info[i]
            ds_b = datasets_info[j]

            # Only compare if they share features
            feats_a = set(ds_a["feature_names"])
            feats_b = set(ds_b["feature_names"])
            shared  = sorted(feats_a & feats_b)
            if len(shared) < 3:
                continue

            for va in ds_a["top_voxels"][:5]:
                for vb in ds_b["top_voxels"][:5]:
                    sim = _centroid_similarity(va["centroid_sc"], vb["centroid_sc"])
                    if sim >= similarity_threshold:
                        # Determine which dataset has the worse gap
                        # (higher gap_score = more underrepresented)
                        if va["gap_score"] >= vb["gap_score"]:
                            sparse_ds, dense_ds = ds_a, ds_b
                            sparse_vox, dense_vox = va, vb
                        else:
                            sparse_ds, dense_ds = ds_b, ds_a
                            sparse_vox, dense_vox = vb, va

                        matches.append({
                            "sparse_dataset": sparse_ds["name"],
                            "dense_dataset":  dense_ds["name"],
                            "sparse_voxel":   sparse_vox["label"],
                            "dense_voxel":    dense_vox["label"],
                            "similarity":     round(sim, 4),
                            "sparse_gap_score": sparse_vox["gap_score"],
                            "shared_features": shared,
                            "_sparse_ds": sparse_ds,
                            "_dense_ds":  dense_ds,
                            "_sparse_vox": sparse_vox,
                            "_dense_vox":  dense_vox,
                        })

    # Deduplicate: keep highest similarity match per (sparse_dataset, sparse_voxel)
    seen = {}
    for m in sorted(matches, key=lambda x: x["similarity"], reverse=True):
        key = (m["sparse_dataset"], m["sparse_voxel"])
        if key not in seen:
            seen[key] = m
    return list(seen.values())


def transfer_samples(matches, n_transfer=20, verbose=True):
    """
    For each matched gap pair, take real samples from the dense dataset
    and reproject them into the sparse dataset's feature space.

    This is strictly better than synthesizing from scratch because:
    - They are REAL samples (not synthetic)
    - They come from the same clinical/demographic region
    - They pass no discriminator test (they ARE real data)

    Parameters
    ----------
    matches    : list of match dicts from find_matching_gaps()
    n_transfer : int — max samples to transfer per match
    verbose    : bool

    Returns
    -------
    transfer_results : dict mapping dataset name -> list of transferred sample arrays
    """
    transfer_results = {}

    for m in matches:
        ds_dense   = m["_dense_ds"]
        ds_sparse  = m["_sparse_ds"]
        dense_vox  = m["_dense_vox"]
        shared     = m["shared_features"]

        dense_X_sc  = ds_dense["X_tr_sc"]
        sparse_X_sc = ds_sparse["X_tr_sc"]
        dense_y     = ds_dense["y_tr"]

        if hasattr(dense_y, "values"):
            dense_y = dense_y.values

        # Get real positive samples from the dense voxel
        vox_indices = dense_vox["indices"]
        vox_y = dense_y[vox_indices]
        pos_indices = vox_indices[vox_y == 1]

        if len(pos_indices) == 0:
            if verbose:
                print("  [TRANSFER] No positive samples in dense voxel " + dense_vox["label"])
            continue

        # Take up to n_transfer real samples
        take = min(n_transfer, len(pos_indices))
        selected = pos_indices[:take]
        samples_sc = dense_X_sc[selected]  # shape (take, n_features_dense)

        # We can only transfer shared feature dimensions
        # Get feature indices for shared features in each dataset
        dense_feat_idx  = [ds_dense["feature_names"].index(f)
                           for f in shared if f in ds_dense["feature_names"]]
        sparse_feat_idx = [ds_sparse["feature_names"].index(f)
                           for f in shared if f in ds_sparse["feature_names"]]

        if len(dense_feat_idx) == 0:
            continue

        shared_samples = samples_sc[:, dense_feat_idx]

        # Re-scale: inverse transform from dense scaler, then scale with sparse scaler
        try:
            dense_scaler  = ds_dense["scaler"]
            sparse_scaler = ds_sparse["scaler"]

            # Get the shared feature sub-scalers
            dense_mean  = dense_scaler.mean_[dense_feat_idx]
            dense_scale = dense_scaler.scale_[dense_feat_idx]
            sparse_mean = sparse_scaler.mean_[sparse_feat_idx]
            sparse_scale= sparse_scaler.scale_[sparse_feat_idx]

            # Inverse transform from dense space
            raw = shared_samples * dense_scale + dense_mean
            # Scale into sparse space
            rescaled = (raw - sparse_mean) / sparse_scale

            # Build full sparse-dimension sample (fill non-shared dims with gap centroid)
            n_sparse_feats = sparse_X_sc.shape[1]
            transferred    = np.tile(ds_sparse["_sparse_vox"]["centroid_sc"], (take, 1))
            for idx_in_shared, sparse_idx in enumerate(sparse_feat_idx):
                transferred[:, sparse_idx] = rescaled[:, idx_in_shared]

            name = ds_sparse["name"]
            if name not in transfer_results:
                transfer_results[name] = []
            transfer_results[name].append(transferred)

            if verbose:
                print("  [TRANSFER] " + str(take) + " real samples from " +
                      ds_dense["name"] + " -> " + name +
                      " (gap=" + dense_vox["label"] + " sim=" + str(m["similarity"]) + ")")

        except Exception as e:
            if verbose:
                print("  [TRANSFER] Failed for " + m["sparse_dataset"] + ": " + str(e))

    return transfer_results


def apply_transfers(dataset_name, X_tr_sc, y_tr, transfer_results):
    """
    Inject transferred real samples into a dataset's training set.

    Returns
    -------
    X_aug : np.ndarray
    y_aug : np.ndarray
    n_added : int
    """
    y_arr = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)

    if dataset_name not in transfer_results:
        return X_tr_sc.copy(), y_arr.copy(), 0

    parts = transfer_results[dataset_name]
    if not parts:
        return X_tr_sc.copy(), y_arr.copy(), 0

    transferred = np.vstack(parts)
    X_aug = np.vstack([X_tr_sc, transferred])
    y_aug = np.concatenate([y_arr, np.ones(len(transferred), dtype=int)])
    return X_aug, y_aug, len(transferred)


def print_transfer_summary(matches, transfer_results):
    """Print a readable summary of what was transferred."""
    print("\n" + "─" * 55)
    print("  MULTI-DATASET GAP TRANSFER SUMMARY")
    print("─" * 55)

    if not matches:
        print("  No matching gap regions found across datasets.")
        print("─" * 55 + "\n")
        return

    print("  Matching gap pairs found: " + str(len(matches)))
    print()
    for m in matches:
        print("  " + m["dense_dataset"] + " -> " + m["sparse_dataset"])
        print("    Voxels : " + m["dense_voxel"] + " -> " + m["sparse_voxel"])
        print("    Similarity : " + str(m["similarity"]))
        print("    Sparse gap score : " + str(m["sparse_gap_score"]))
        print("    Shared features  : " + str(len(m["shared_features"])))
        print()

    print("  Transferred samples:")
    for name, parts in transfer_results.items():
        n = sum(len(p) for p in parts)
        print("    " + name + ": +" + str(n) + " real samples injected")

    print("─" * 55 + "\n")
