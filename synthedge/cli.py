"""
synthedge.cli
=============
Command-line interface for SynthEdge.

Commands
--------
    synthedge analyze  data.csv --target diagnosis
    synthedge fill     data.csv --target diagnosis --n-top 3 --out augmented.csv
    synthedge report   data.csv --target diagnosis --out report.html
    synthedge compare  data.csv --target diagnosis
    synthedge transfer dataset_a.csv dataset_b.csv --target diagnosis
"""

import argparse
import sys
import os
import time
import pandas as pd
from . import SynthEdge


# ── Helpers ───────────────────────────────────────────────────────────────

def _load(path, target):
    if not os.path.exists(path):
        print("Error: file not found — " + path)
        sys.exit(1)
    df = pd.read_csv(path)
    if target not in df.columns:
        print("Error: column '" + target + "' not found in " + path)
        print("Available columns: " + str(list(df.columns)))
        sys.exit(1)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    df[target] = (df[target] > 0).astype(int)
    return df


def _banner(text):
    print()
    print("  " + "─" * 52)
    print("  " + text)
    print("  " + "─" * 52)


# ── Command: analyze ──────────────────────────────────────────────────────

def cmd_analyze(args):
    _banner("SynthEdge  analyze")
    print("  File   : " + args.file)
    print("  Target : " + args.target)

    df = _load(args.file, args.target)
    print("  Rows   : " + str(len(df)) +
          "  Positive rate: " + str(round(df[args.target].mean()*100,1)) + "%")

    se = SynthEdge(df, target_col=args.target, verbose=True)
    se.analyze()

    print()
    print("  Gap voxel map (top 10):")
    print(se.gap_map.to_string(index=False))
    print()
    print("  Run 'synthedge fill " + args.file +
          " --target " + args.target + "' to augment.")


# ── Command: fill ─────────────────────────────────────────────────────────

def cmd_fill(args):
    _banner("SynthEdge  fill")
    print("  File   : " + args.file)
    print("  Target : " + args.target)

    df   = _load(args.file, args.target)
    out  = args.out or args.file.replace(".csv", "_augmented.csv")

    se   = SynthEdge(df, target_col=args.target, verbose=True)
    se.analyze()
    aug  = se.fill(
        n_top        = args.n_top,
        ctgan_epochs = args.epochs,
        use_ctgan    = not args.no_ctgan,
    )

    aug.to_csv(out, index=False)
    q = se.quality_report()

    print()
    print("  Results:")
    print("    Original rows  : " + str(len(df)))
    print("    Augmented rows : " + str(len(aug)))
    print("    Samples added  : +" + str(q["total_added"]))
    print("    Method         : " + ("CTGAN" if q["ctgan_used"] else "Gaussian"))
    print("    Drift features : " + str(len(q["drift_features"])))
    print()
    print("  Saved to: " + out)

    if args.report:
        rpath = out.replace(".csv", "_report.html")
        se.save_report(output_path=rpath, dataset_name=os.path.basename(args.file))
        print("  Report : " + rpath)


# ── Command: report ───────────────────────────────────────────────────────

def cmd_report(args):
    _banner("SynthEdge  report")
    print("  File   : " + args.file)
    print("  Target : " + args.target)
    print("  Output : " + args.out)

    df = _load(args.file, args.target)
    se = SynthEdge(df, target_col=args.target, verbose=True)
    se.analyze()

    # Run fill so synthesis metadata is populated
    se.fill(
        n_top        = args.n_top,
        ctgan_epochs = args.epochs,
        use_ctgan    = not args.no_ctgan,
    )

    # Parse optional comparison JSON if provided
    comparison = None
    if args.comparison:
        import json
        try:
            comparison = json.loads(args.comparison)
        except Exception as e:
            print("  Warning: could not parse --comparison JSON (" + str(e) + ")")

    path = se.save_report(
        output_path      = args.out,
        dataset_name     = args.name or os.path.basename(args.file).replace(".csv",""),
        comparison_results = comparison,
    )
    print()
    print("  Report saved: " + path)
    print("  Open in any browser to view.")


# ── Command: compare ──────────────────────────────────────────────────────

def cmd_compare(args):
    _banner("SynthEdge  compare")
    print("  File   : " + args.file)
    print("  Target : " + args.target)
    print()

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (recall_score, f1_score,
                                  roc_auc_score, average_precision_score,
                                  classification_report)
    from imblearn.over_sampling import SMOTE
    import numpy as np

    try:
        from xgboost import XGBClassifier
        def get_clf(pos_w):
            return XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                scale_pos_weight=pos_w, eval_metric="logloss",
                random_state=42, verbosity=0)
        model_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        def get_clf(pos_w):
            return RandomForestClassifier(
                n_estimators=150, class_weight="balanced", random_state=42)
        model_name = "RandomForest"

    df = _load(args.file, args.target)
    print("  Dataset: " + str(len(df)) + " rows, " +
          str(round(df[args.target].mean()*100,1)) + "% positive")
    print("  Model  : " + model_name)
    print()

    X = df[[c for c in df.columns if c != args.target]]
    y = df[args.target]

    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    for o in [X_tr_df, X_te_df, y_tr, y_te]:
        o.reset_index(drop=True, inplace=True)

    sc      = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_df)
    X_te_sc = sc.transform(X_te_df)
    pos_w   = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1)

    def evaluate(X_t, y_t, label):
        t0  = time.time()
        clf = get_clf(pos_w)
        clf.fit(X_t, y_t)
        y_pred  = clf.predict(X_te_sc)
        y_proba = clf.predict_proba(X_te_sc)[:,1]
        elapsed = round(time.time()-t0, 1)
        rec  = round(recall_score(y_te, y_pred, zero_division=0), 4)
        f1   = round(f1_score(y_te, y_pred, zero_division=0), 4)
        auc  = round(roc_auc_score(y_te, y_proba), 4)
        pr   = round(average_precision_score(y_te, y_proba), 4)
        print("  " + label.ljust(20) +
              "  Recall=" + str(rec) +
              "  F1=" + str(f1) +
              "  ROC-AUC=" + str(auc) +
              "  PR-AUC=" + str(pr) +
              "  (" + str(elapsed) + "s)")
        return {"recall": rec, "f1": f1, "roc_auc": auc, "pr_auc": pr}

    print("  " + "─"*60)
    print("  " + "Method".ljust(20) +
          "  Recall    F1        ROC-AUC   PR-AUC")
    print("  " + "─"*60)

    # No augmentation
    r_base = evaluate(X_tr_sc, y_tr.values, "No augmentation")

    # SMOTE
    k = max(1, min(5, int(y_tr.sum())-1))
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_sm, y_sm = sm.fit_resample(X_tr_sc, y_tr)
    r_sm   = evaluate(X_sm, y_sm, "SMOTE (+"+str(int(y_sm.sum())-int(y_tr.sum()))+")")

    # SynthEdge
    se = SynthEdge(df[[c for c in df.columns if c != args.target]
                      ].assign(**{args.target: df[args.target]}),
                   target_col=args.target, verbose=False)
    aug_df  = se.fill(n_top=args.n_top, ctgan_epochs=args.epochs,
                      use_ctgan=not args.no_ctgan)
    X_se_sc = sc.transform(aug_df[[c for c in df.columns if c != args.target]])
    y_se    = aug_df[args.target].values
    se_added = int(y_se.sum()) - int(y_tr.sum())
    r_se    = evaluate(X_se_sc, y_se,
                       "SynthEdge (+" + str(se_added) + ")")

    print("  " + "─"*60)
    print()

    # Summary
    gains = {
        "Recall":  round((r_se["recall"]  - r_sm["recall"])  * 100, 1),
        "F1":      round((r_se["f1"]      - r_sm["f1"])      * 100, 1),
        "ROC-AUC": round((r_se["roc_auc"] - r_sm["roc_auc"]) * 100, 1),
        "PR-AUC":  round((r_se["pr_auc"]  - r_sm["pr_auc"])  * 100, 1),
    }
    print("  SynthEdge vs SMOTE:")
    for metric, gain in gains.items():
        flag = "  WIN" if gain > 0.1 else ("  LOSS" if gain < -0.1 else "  TIE")
        print("    " + metric.ljust(10) +
              ("+"+str(gain) if gain >= 0 else str(gain)) + " pp" + flag)

    print()
    print("  Severity: " + se.severity["severity"] +
          "  |  Will help: " + str(se.severity["will_help"]))
    print()

    # Optionally save report
    if args.report:
        rpath = args.file.replace(".csv", "_compare_report.html")
        comparison = {
            "No augmentation": r_base,
            "SMOTE":           r_sm,
            "SynthEdge":       r_se,
        }
        se.save_report(
            output_path      = rpath,
            dataset_name     = os.path.basename(args.file).replace(".csv",""),
            comparison_results = comparison,
        )
        print("  Report saved: " + rpath)


# ── Command: transfer ─────────────────────────────────────────────────────

def cmd_transfer(args):
    _banner("SynthEdge  transfer")
    print("  Source (dense) : " + args.source)
    print("  Target (sparse): " + args.target_file)
    print("  Target column  : " + args.target)
    print()

    from sklearn.preprocessing import StandardScaler
    from .scanner import scan
    from .transfer import (find_matching_gaps, transfer_samples,
                            apply_transfers, print_transfer_summary)

    df_src = _load(args.source,      args.target)
    df_tgt = _load(args.target_file, args.target)

    feats_src = [c for c in df_src.columns if c != args.target]
    feats_tgt = [c for c in df_tgt.columns if c != args.target]
    shared    = sorted(set(feats_src) & set(feats_tgt))

    print("  Source : " + str(len(df_src)) + " rows  (" +
          str(round(df_src[args.target].mean()*100,1)) + "% positive)")
    print("  Target : " + str(len(df_tgt)) + " rows  (" +
          str(round(df_tgt[args.target].mean()*100,1)) + "% positive)")
    print("  Shared features: " + str(len(shared)))

    if len(shared) < 3:
        print()
        print("  Warning: fewer than 3 shared features — transfer may be unreliable.")

    sc_src = StandardScaler()
    sc_tgt = StandardScaler()
    X_src  = sc_src.fit_transform(df_src[feats_src])
    X_tgt  = sc_tgt.fit_transform(df_tgt[feats_tgt])
    y_src  = df_src[args.target]
    y_tgt  = df_tgt[args.target]

    top_src, _, _ = scan(X_src, y_src, top_k=5)
    top_tgt, _, _ = scan(X_tgt, y_tgt, top_k=5)

    datasets_info = [
        {"name": os.path.basename(args.source),
         "top_voxels": top_src, "scaler": sc_src,
         "X_tr_sc": X_src, "y_tr": y_src,
         "feature_names": feats_src,
         "_sparse_vox": top_src[0] if top_src else None},
        {"name": os.path.basename(args.target_file),
         "top_voxels": top_tgt, "scaler": sc_tgt,
         "X_tr_sc": X_tgt, "y_tr": y_tgt,
         "feature_names": feats_tgt,
         "_sparse_vox": top_tgt[0] if top_tgt else None},
    ]

    matches   = find_matching_gaps(datasets_info,
                                    similarity_threshold=args.threshold)
    transfers = transfer_samples(matches, n_transfer=args.n_transfer, verbose=True)
    print_transfer_summary(matches, transfers)

    tgt_name = os.path.basename(args.target_file)
    X_aug, y_aug, n_added = apply_transfers(tgt_name, X_tgt, y_tgt, transfers)

    if n_added > 0:
        # Inverse transform back to original scale and save
        import numpy as np
        X_raw   = sc_tgt.inverse_transform(X_aug)
        aug_df  = pd.DataFrame(X_raw, columns=feats_tgt)
        aug_df[args.target] = y_aug
        out = args.out or args.target_file.replace(".csv", "_transferred.csv")
        aug_df.to_csv(out, index=False)
        print("  Added " + str(n_added) + " real samples from " +
              os.path.basename(args.source))
        print("  Saved to: " + out)
    else:
        print("  No matching gaps found at threshold=" + str(args.threshold))
        print("  Try lowering --threshold (current: " + str(args.threshold) + ")")


# ── Main parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog        = "synthedge",
        description = "SynthEdge — diagnosis-first synthetic data augmentation",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
commands:
  analyze   Detect gap regions and severity in a dataset
  fill      Augment dataset by filling identified gap regions
  report    Generate a standalone HTML gap report
  compare   Run full SMOTE vs SynthEdge benchmark comparison
  transfer  Transfer real samples from one dataset to fill gaps in another

examples:
  synthedge analyze  data.csv --target diagnosis
  synthedge fill     data.csv --target diagnosis --n-top 3 --out aug.csv
  synthedge report   data.csv --target diagnosis --out report.html
  synthedge compare  data.csv --target diagnosis --report
  synthedge transfer source.csv target.csv --target diagnosis --threshold 0.6
        """
    )
    sub = parser.add_subparsers(dest="command")

    # ── analyze ──────────────────────────────────────────────────────
    p = sub.add_parser("analyze", help="Detect gap regions and severity")
    p.add_argument("file",     help="Path to CSV file")
    p.add_argument("--target", default="target", help="Target column (default: target)")

    # ── fill ─────────────────────────────────────────────────────────
    p = sub.add_parser("fill", help="Augment dataset by filling gap regions")
    p.add_argument("file",       help="Path to CSV file")
    p.add_argument("--target",   default="target", help="Target column")
    p.add_argument("--n-top",    type=int, default=3,   help="Top N voxels to fill (default: 3)")
    p.add_argument("--epochs",   type=int, default=100, help="CTGAN epochs (default: 100)")
    p.add_argument("--no-ctgan", action="store_true",   help="Use Gaussian only (faster, no CTGAN)")
    p.add_argument("--out",      default=None,          help="Output CSV path")
    p.add_argument("--report",   action="store_true",   help="Also save HTML gap report")

    # ── report ───────────────────────────────────────────────────────
    p = sub.add_parser("report", help="Generate standalone HTML gap report")
    p.add_argument("file",         help="Path to CSV file")
    p.add_argument("--target",     default="target",               help="Target column")
    p.add_argument("--out",        default="synthedge_report.html",help="Output HTML path")
    p.add_argument("--name",       default=None,   help="Dataset display name in report")
    p.add_argument("--n-top",      type=int, default=3,   help="Top N voxels (default: 3)")
    p.add_argument("--epochs",     type=int, default=50,  help="CTGAN epochs (default: 50)")
    p.add_argument("--no-ctgan",   action="store_true",   help="Gaussian only")
    p.add_argument("--comparison", default=None,
                   help='JSON string of comparison results, e.g. \'{"SMOTE":{"recall":0.3,"f1":0.28,"roc_auc":0.7,"pr_auc":0.5}}\'')

    # ── compare ──────────────────────────────────────────────────────
    p = sub.add_parser("compare", help="Full SMOTE vs SynthEdge benchmark")
    p.add_argument("file",       help="Path to CSV file")
    p.add_argument("--target",   default="target", help="Target column")
    p.add_argument("--n-top",    type=int, default=3,   help="Top N voxels (default: 3)")
    p.add_argument("--epochs",   type=int, default=50,  help="CTGAN epochs (default: 50)")
    p.add_argument("--no-ctgan", action="store_true",   help="Gaussian only")
    p.add_argument("--report",   action="store_true",   help="Save HTML report with charts")

    # ── transfer ─────────────────────────────────────────────────────
    p = sub.add_parser("transfer", help="Transfer real samples between datasets")
    p.add_argument("source",       help="Source (dense) dataset CSV")
    p.add_argument("target_file",  help="Target (sparse) dataset CSV to augment")
    p.add_argument("--target",     default="target", help="Target column (same in both files)")
    p.add_argument("--threshold",  type=float, default=0.65,
                   help="Cosine similarity threshold for gap matching (default: 0.65)")
    p.add_argument("--n-transfer", type=int, default=20,
                   help="Max samples to transfer per match (default: 20)")
    p.add_argument("--out",        default=None, help="Output CSV path")

    # ── dispatch ─────────────────────────────────────────────────────
    args = parser.parse_args()

    dispatch = {
        "analyze":  cmd_analyze,
        "fill":     cmd_fill,
        "report":   cmd_report,
        "compare":  cmd_compare,
        "transfer": cmd_transfer,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
